/**
 * zyroute — one gateway, both dialects, in front of AWS Bedrock Mantle.
 *
 *   POST /v1/chat/completions      OpenAI dialect
 *   POST /v1/messages              Anthropic dialect (Claude Code)
 *   POST /v1/messages/count_tokens Anthropic dialect
 *   POST /v1/responses             OpenAI Responses dialect (passthrough)
 *   GET  /v1/models                live catalogue, proxied from Bedrock
 *   GET  /v1/models/:id
 *   GET  /health
 *
 * Mantle exposes two upstream surfaces and the model id decides which one is used:
 *   anthropic.claude-*  -> /anthropic/v1/messages           (SigV4 or bearer)
 *   everything else     -> /v1/chat/completions, /v1/responses  (bearer only)
 * Either client dialect reaches either surface; the gateway translates when the two
 * don't line up.
 *
 * Note the upstream paths: Mantle's OpenAI-compatible surface for its own catalogue
 * lives at the host ROOT (/v1/chat/completions). The host also answers on /openai/v1/*,
 * but that is a different, older surface that does not serve the Mantle catalogue —
 * openai.gpt-oss-20b there returns "isn't supported on this route".
 *
 * Point Claude Code at it with ANTHROPIC_BASE_URL — no path suffix, the gateway
 * serves /v1/* at the root. The /anthropic prefix belongs to the upstream Mantle
 * URL only; clients never see it.
 *
 * Upstream env — all read by the Mantle client itself, in this precedence:
 *   AWS_BEARER_TOKEN_BEDROCK          Bedrock API key. Sent as `Authorization: Bearer`,
 *                                     no SigV4, no access key/secret needed. Required
 *                                     for the OpenAI surface.
 *   AWS_ACCESS_KEY_ID / _SECRET_ACCESS_KEY / _SESSION_TOKEN, AWS_PROFILE
 *                                     SigV4 fallback via the default AWS chain.
 *   AWS_REGION (or AWS_DEFAULT_REGION)
 *                                     Builds the default endpoint, and is required
 *                                     for SigV4 signing. Optional if the base URL
 *                                     is set explicitly and a bearer token is used.
 *   ANTHROPIC_BEDROCK_MANTLE_BASE_URL Overrides the endpoint. With or without the
 *                                     /anthropic suffix; both are accepted.
 *
 * Gateway env (loaded from .env by the npm scripts; a variable already exported in
 * the shell wins over .env — that is Node's --env-file behaviour, not ours):
 *   GATEWAY_API_KEYS      comma-separated; when unset, auth is skipped and the
 *                         server binds to 127.0.0.1 only
 *   SYSTEM_PROMPT_FILE    defaults to ./system-prompt.md
 *   DEFAULT_MODEL         fallback for ids that aren't Mantle ids
 *   RATE_LIMIT_PER_MINUTE per key, default 120; 0 disables
 *   MAX_BODY_MB           default 32
 *   NO_SAMPLING_MODELS    regex of models that reject temperature/top_p
 *   LOG_BODIES=1          dump every translated upstream payload
 *   PRETTY_JSON=0         turn off indented JSON responses
 *   HOST, PORT
 */
import { serve } from "@hono/node-server";
import Anthropic from "@anthropic-ai/sdk";
import { AnthropicBedrockMantle } from "@anthropic-ai/bedrock-sdk";
import { Hono, type Context } from "hono";
import { streamSSE } from "hono/streaming";
import {
  BASE_SYSTEM,
  DEFAULT_MODEL,
  makeChunker,
  countBreakpoints,
  mergeSystem,
  modelList,
  normalize,
  openAIToAnthropic,
  resolveModel,
  toChatCompletion,
  withBaseSystem,
  type OpenAIRequest,
  type Params,
} from "./translate.ts";
import {
  chatToMessage,
  makeMessageEmitter,
  messagesToChat,
  sseJSON,
  type ChatResponse,
} from "./anthropic-to-openai.ts";
import { bodyLimit } from "hono/body-limit";
import { makeKeyGuard, makeRateLimiter, presentedKey } from "./auth.ts";
import { bold, cyan, dim, logBody, ms, redact, statusColor } from "./log.ts";

/** Accepts the host with or without a /anthropic suffix. */
function mantleHost(): string {
  const raw = process.env.ANTHROPIC_BEDROCK_MANTLE_BASE_URL?.trim().replace(/\/+$/, "");
  if (raw) return raw.replace(/\/anthropic$/, "");
  const region = process.env.AWS_REGION ?? process.env.AWS_DEFAULT_REGION;
  if (!region) throw new Error("set AWS_REGION or ANTHROPIC_BEDROCK_MANTLE_BASE_URL");
  return `https://bedrock-mantle.${region}.api.aws`;
}

const HOST = mantleHost();

const client = new AnthropicBedrockMantle({
  awsRegion: process.env.AWS_REGION,
  baseURL: `${HOST}/anthropic`,
});

const guard = makeKeyGuard(process.env.GATEWAY_API_KEYS);

/** Set by the handlers so the log line can name the model and upstream surface. */
type Vars = { model?: string; upstream?: string; streaming?: boolean };

const app = new Hono<{ Variables: Vars }>();

// One line per request: method, path, status, duration, model, upstream surface.
// On streaming requests the duration is time-to-handler-return (roughly time to first
// byte), not the length of the stream — the handler returns while SSE is still flowing.
app.use("*", async (c, next) => {
  const started = Date.now();
  await next();
  if (c.req.path === "/health") return;
  console.log(
    [
      bold(c.req.method),
      c.req.path,
      statusColor(c.res.status),
      dim(ms(started)),
      c.get("model") ? cyan(c.get("model")!) : "",
      c.get("upstream") ? dim(`→ ${c.get("upstream")}`) : "",
      c.get("streaming") ? dim("stream") : "",
    ]
      .filter(Boolean)
      .join(" "),
  );
});

// Indented JSON — this is a debugging gateway, readable curl output beats a few saved
// bytes. Streams are left alone: buffering an SSE body here would defeat streaming.
const PRETTY = process.env.PRETTY_JSON !== "0";
app.use("*", async (c, next) => {
  await next();
  if (!PRETTY || !c.res.headers.get("content-type")?.includes("application/json")) return;
  const text = await c.res.clone().text();
  try {
    const headers = new Headers(c.res.headers);
    headers.delete("content-length");
    c.res = new Response(JSON.stringify(JSON.parse(text), null, 2), {
      status: c.res.status,
      headers,
    });
  } catch {
    // not JSON after all — leave the original response alone
  }
});

// Harnesses send large contexts and inline images; generous, but not unbounded.
app.use("/v1/*", bodyLimit({ maxSize: Number(process.env.MAX_BODY_MB ?? 32) * 1024 * 1024 }));

const exceeded = makeRateLimiter(Number(process.env.RATE_LIMIT_PER_MINUTE ?? 120));

app.use("/v1/*", async (c, next) => {
  const key = presentedKey(c.req.header("authorization"), c.req.header("x-api-key"));
  if (!guard.accepts(key)) {
    return c.json({ error: { message: "invalid api key", type: "authentication_error" } }, 401);
  }
  if (exceeded(key, Date.now())) {
    return c.json(
      { error: { message: "rate limit exceeded", type: "rate_limit_error" } },
      429,
      { "retry-after": "60" },
    );
  }
  return next();
});

app.get("/health", (c) => c.json({ ok: true, model: DEFAULT_MODEL, upstream: HOST }));

app.get("/v1/models", async (c) => {
  c.set("upstream", "/v1/models");
  return c.json(modelList(await fetchModels(c.req.raw.signal)));
});

app.get("/v1/models/:id", async (c) => {
  const id = c.req.param("id");
  c.set("model", id);
  c.set("upstream", "/v1/models");
  const found = (await fetchModels(c.req.raw.signal)).find((m) => m.id === id);
  if (!found) {
    return c.json(
      { type: "error", error: { type: "not_found_error", message: `model "${id}" not found` } },
      404,
    );
  }
  return c.json(modelList([found]).data[0]!);
});

/** Missing configuration, as opposed to an upstream failure. */
class ConfigError extends Error {}

/**
 * A non-2xx from Mantle, carried out to onError with its status intact.
 *
 * Written without a constructor parameter property on purpose: Node runs this file's
 * TypeScript directly in strip-only mode, and parameter properties need a real
 * transform. Keeping them out is what lets the Docker image skip tsx and any build step.
 */
class UpstreamError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/**
 * Calls Mantle's root (OpenAI-compatible) surface. GET when no payload is given.
 *
 * ponytail: bearer token only. The Anthropic surface gets SigV4 from the Mantle
 * client; signing this one by hand needs @smithy/signature-v4 — add it if you must
 * use IAM credentials instead of AWS_BEARER_TOKEN_BEDROCK.
 */
function mantleFetch(
  path: string,
  payload?: unknown,
  signal?: AbortSignal,
): Promise<Response> {
  const token = process.env.AWS_BEARER_TOKEN_BEDROCK;
  if (!token) {
    throw new ConfigError(
      `${path} is reached over Mantle's bearer-token surface, which needs AWS_BEARER_TOKEN_BEDROCK. Set it, or select an anthropic.* model (that path supports SigV4).`,
    );
  }
  if (payload !== undefined) logBody(`upstream ${path}`, payload);
  return fetch(`${HOST}${path}`, {
    method: payload === undefined ? "GET" : "POST",
    headers: {
      authorization: `Bearer ${token}`,
      ...(payload !== undefined && { "content-type": "application/json" }),
    },
    ...(payload !== undefined && { body: JSON.stringify(payload) }),
    signal,
  });
}

/** Upstream model catalogue. Never cached, never hardcoded. */
async function fetchModels(signal: AbortSignal) {
  const upstream = await mantleFetch("/v1/models", undefined, signal);
  if (!upstream.ok) {
    throw new UpstreamError(upstream.status || 502, await upstream.text());
  }
  const body = (await upstream.json()) as {
    data?: { id: string; created?: number; owned_by?: string }[];
  };
  return body.data ?? [];
}

/**
 * Pipes an SSE body through byte-for-byte, appending an in-band error frame if the
 * upstream stream breaks. Without it a broken stream just stops, and the client has a
 * 200 with no terminator to explain it.
 */
function guardedSSE(body: ReadableStream<Uint8Array>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    async start(controller) {
      const reader = body.getReader();
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          controller.enqueue(value);
        }
      } catch (err) {
        const message = redact(err instanceof Error ? err.message : String(err));
        console.error(`stream aborted: ${message}`);
        const frame = JSON.stringify({ error: { message, type: "api_error" } });
        controller.enqueue(encoder.encode(`event: error\ndata: ${frame}\n\n`));
      }
      controller.close();
    },
  });
}

/** Forwards an upstream response, untouched apart from the stream-failure guard. */
function pipeThrough(upstream: Response): Response {
  const type = upstream.headers.get("content-type") ?? "application/json";
  const streaming = type.includes("event-stream");
  return new Response(
    streaming && upstream.body ? guardedSSE(upstream.body) : upstream.body,
    {
      status: upstream.status,
      headers: {
        "content-type": type,
        ...(streaming && { "cache-control": "no-cache", connection: "keep-alive" }),
      },
    },
  );
}

/**
 * A stream that dies after the headers are sent cannot become an HTTP error code, so
 * it has to end with an in-band error event. Without one the client sits waiting for
 * a message_stop that never arrives — Anthropic's own API emits `event: error` here,
 * so clients already know the shape.
 */
async function sseFailure(
  sse: { writeSSE: (m: { event?: string; data: string }) => Promise<void> },
  dialect: "anthropic" | "openai",
  err: unknown,
) {
  const message = redact(err instanceof Error ? err.message : String(err));
  console.error(`stream aborted: ${message}`);
  try {
    if (dialect === "anthropic") {
      await sse.writeSSE({
        event: "error",
        data: JSON.stringify({ type: "error", error: { type: "api_error", message } }),
      });
    } else {
      await sse.writeSSE({ data: JSON.stringify({ error: { message, type: "api_error" } }) });
      await sse.writeSSE({ data: "[DONE]" });
    }
  } catch {
    // client is already gone; nothing left to tell it
  }
}

/** Streams raw Anthropic events straight back to the client. */
function relayAnthropic(c: Context, params: Params) {
  return streamSSE(c, async (sse) => {
    const upstream = client.messages.stream(params);
    sse.onAbort(() => upstream.abort());
    try {
      for await (const ev of upstream) {
        await sse.writeSSE({ event: ev.type, data: JSON.stringify(ev) });
      }
    } catch (err) {
      await sseFailure(sse, "anthropic", err);
    }
  });
}

/** Anthropic-dialect client, non-Anthropic model: translate both ways. */
async function bridgeToChat(
  c: Context,
  p: Anthropic.MessageCreateParams,
  model: string,
  streaming: boolean,
) {
  const shown = p.model; // echo the client's own id back
  const payload = messagesToChat(
    {
      ...p,
      // no cache_control on the OpenAI surface, so the breakpoint cap is irrelevant
      system: mergeSystem(p.system as string | Anthropic.TextBlockParam[] | undefined, false),
    },
    model,
    streaming,
  );
  const upstream = await mantleFetch("/v1/chat/completions", payload, c.req.raw.signal);

  if (!upstream.ok) {
    return c.json(
      { type: "error", error: { type: "api_error", message: redact(await upstream.text()) } },
      (upstream.status || 500) as 500,
    );
  }

  if (!streaming) {
    const completion = (await upstream.json()) as ChatResponse;
    logBody("upstream response", completion);
    return c.json(chatToMessage(completion, shown));
  }

  const body = upstream.body;
  if (!body) throw new UpstreamError(502, "upstream returned no body for a streaming request");

  return streamSSE(c, async (sse) => {
    const emitter = makeMessageEmitter(shown);
    try {
      for await (const chunk of sseJSON(body)) {
        for (const ev of emitter.push(chunk)) {
          await sse.writeSSE({ event: ev.type, data: JSON.stringify(ev) });
        }
      }
      for (const ev of emitter.finish()) {
        await sse.writeSSE({ event: ev.type, data: JSON.stringify(ev) });
      }
    } catch (err) {
      await sseFailure(sse, "anthropic", err);
    }
  });
}

app.post("/v1/messages", async (c) => {
  const { stream, ...rest } = await c.req.json<Anthropic.MessageCreateParams>();
  const { model, family } = resolveModel(rest.model);
  c.set("model", model);
  c.set("streaming", !!stream);

  if (family === "openai") {
    c.set("upstream", "/v1/chat/completions");
    return bridgeToChat(c, rest as Anthropic.MessageCreateParams, model, !!stream);
  }

  c.set("upstream", "/anthropic/v1/messages");
  const params = normalize(rest as Params, !!stream);
  logBody("upstream /anthropic/v1/messages", params);
  if (stream) return relayAnthropic(c, params);
  return c.json(await client.messages.create({ ...params, stream: false }));
});

// Claude Code calls this before long requests; a 404 here makes it fall back to a
// rough local estimate. Counts the merged system prompt, not just the client's.
app.post("/v1/messages/count_tokens", async (c) => {
  const body = await c.req.json<Anthropic.MessageCountTokensParams>();
  const { model } = resolveModel(body.model);
  c.set("model", model);
  c.set("upstream", "/anthropic/v1/messages/count_tokens");
  return c.json(
    await client.messages.countTokens({
      ...body,
      model,
      system: mergeSystem(
        body.system as string | Anthropic.TextBlockParam[] | undefined,
        countBreakpoints(body) < 4,
      ),
    }),
  );
});

app.post("/v1/chat/completions", async (c) => {
  const body = await c.req.json<OpenAIRequest>();
  const { model, family } = resolveModel(body.model);
  c.set("model", model);
  c.set("streaming", !!body.stream);

  // Same dialect on both ends — inject the base prompt, fix the id, forward the bytes.
  if (family === "openai") {
    c.set("upstream", "/v1/chat/completions");
    const payload = { ...body, model, messages: withBaseSystem(body.messages ?? []) };
    return pipeThrough(
      await mantleFetch("/v1/chat/completions", payload, c.req.raw.signal),
    );
  }

  c.set("upstream", "/anthropic/v1/messages");
  const params = normalize(openAIToAnthropic(body), !!body.stream);
  const shown = body.model ?? params.model;
  logBody("upstream /anthropic/v1/messages", params);

  if (!body.stream) {
    const msg = await client.messages.create({ ...params, stream: false });
    return c.json(toChatCompletion(msg, shown));
  }

  return streamSSE(c, async (sse) => {
    const upstream = client.messages.stream(params);
    sse.onAbort(() => upstream.abort());
    const toChunks = makeChunker(shown);
    try {
      for await (const ev of upstream) {
        for (const chunk of toChunks(ev)) await sse.writeSSE({ data: JSON.stringify(chunk) });
      }
      await sse.writeSSE({ data: "[DONE]" });
    } catch (err) {
      await sseFailure(sse, "openai", err);
    }
  });
});

// Responses API is OpenAI-surface only. The base prompt rides in `instructions`, with
// the client's own instructions appended behind it.
app.post("/v1/responses", async (c) => {
  const body = await c.req.json<Record<string, unknown>>();
  const { model } = resolveModel(typeof body.model === "string" ? body.model : undefined);
  c.set("model", model);
  c.set("upstream", "/v1/responses");
  c.set("streaming", body.stream === true);

  const instructions = [BASE_SYSTEM, typeof body.instructions === "string" ? body.instructions : ""]
    .filter(Boolean)
    .join("\n\n");

  return pipeThrough(
    await mantleFetch("/v1/responses", { ...body, model, instructions }, c.req.raw.signal),
  );
});

app.onError((err, c) => {
  if (err instanceof UpstreamError) {
    console.error(`upstream ${err.status}: ${err.message}`);
    return c.json(
      { error: { message: redact(err.message), type: "upstream_error" } },
      err.status as 500,
    );
  }
  if (err instanceof ConfigError) {
    console.error(`config: ${err.message}`);
    return c.json({ error: { message: err.message, type: "configuration_error" } }, 501);
  }
  if (err instanceof Anthropic.APIError) {
    console.error(`bedrock ${err.status}: ${err.message}`);
    return c.json(
      { error: { message: redact(err.message), type: err.name, code: err.status ?? null } },
      (err.status ?? 500) as 500,
    );
  }
  console.error(err);
  return c.json({ error: { message: redact(String(err)) } }, 500);
});

const port = Number(process.env.PORT ?? 8787);
const hostname = process.env.HOST ?? (guard.required ? "0.0.0.0" : "127.0.0.1");
const loopback = ["127.0.0.1", "localhost", "::1"].includes(hostname);

// An unauthenticated gateway on a public interface is an open door to your Bedrock
// billing. A warning is not enough when HOST was set explicitly.
if (!guard.required && !loopback) {
  throw new Error(
    `refusing to bind ${hostname} with GATEWAY_API_KEYS unset — anyone who can reach this port could spend from your AWS account. Set GATEWAY_API_KEYS, or bind 127.0.0.1.`,
  );
}
if (!guard.required) {
  console.warn("GATEWAY_API_KEYS unset — no auth, bound to 127.0.0.1. Set it before exposing.");
}
serve({ fetch: app.fetch, port, hostname });
console.log(
  `zyroute on http://${hostname}:${port} → ${HOST} ` +
    `(${process.env.AWS_BEARER_TOKEN_BEDROCK ? "bearer token" : "sigv4, OpenAI surface unavailable"})`,
);

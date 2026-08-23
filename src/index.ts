/**
 * zyroute — one gateway, both dialects, in front of AWS Bedrock.
 *
 *   POST /v1/chat/completions      OpenAI dialect
 *   POST /v1/messages              Anthropic dialect (Claude Code)
 *   POST /v1/messages/count_tokens Anthropic dialect
 *   GET  /v1/models                live foundation-model catalogue
 *   GET  /v1/models/:id
 *   GET  /health
 *
 * One upstream: Bedrock Runtime's Converse API, reaching every foundation model the
 * account can call. Bedrock Mantle was the original target and has been dropped — it
 * carried roughly half the catalogue, lacked Claude Opus 4.6 entirely, and needed three
 * upstream surfaces to cover.
 *
 * Auth is a single bearer token (a Bedrock API key). No IAM credentials, no SigV4, no AWS
 * SDK — the whole thing is plain HTTP against two hosts:
 *
 *   {control}/foundation-models, /inference-profiles     the catalogue
 *   {runtime}/model/{id}/converse | converse-stream      inference
 *   {runtime}/model/{id}/count-tokens                    token counting
 *
 * Both client dialects converge on one internal shape (Anthropic Messages) before being
 * translated to Converse once, so there is no second translator to keep in step.
 *
 * Env: see README.md and .env.example.
 */
import { serve } from "@hono/node-server";
import { Hono, type Context } from "hono";
import { bodyLimit } from "hono/body-limit";
import { streamSSE } from "hono/streaming";
import type Anthropic from "@anthropic-ai/sdk";
import {
  DEFAULT_MODEL,
  countBreakpoints,
  makeChunker,
  mergeSystem,
  modelList,
  normalize,
  openAIToAnthropic,
  resolveModel,
  toChatCompletion,
  type OpenAIRequest,
} from "./translate.ts";
import {
  converseToMessage,
  makeConverseEmitter,
  messagesToConverse,
  type ConverseResponse,
} from "./converse.ts";
import { eventStreamFrames } from "./eventstream.ts";
import { makeCatalogue, type Entry } from "./catalogue.ts";
import { makeKeyGuard, makeRateLimiter, presentedKey } from "./auth.ts";
import { bold, cyan, dim, logBody, ms, redact, statusColor } from "./log.ts";

const REGION = process.env.AWS_REGION ?? process.env.AWS_DEFAULT_REGION ?? "";
if (!REGION && !process.env.BEDROCK_RUNTIME_URL) {
  throw new Error("set AWS_REGION (or AWS_DEFAULT_REGION)");
}
const RUNTIME = process.env.BEDROCK_RUNTIME_URL ?? `https://bedrock-runtime.${REGION}.amazonaws.com`;
const CONTROL = process.env.BEDROCK_CONTROL_URL ?? `https://bedrock.${REGION}.amazonaws.com`;

const guard = makeKeyGuard(process.env.GATEWAY_API_KEYS);

/** Missing configuration, as opposed to an upstream failure. */
class ConfigError extends Error {}

/** A non-2xx from Bedrock, carried out to onError with its status intact. */
class UpstreamError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** POST when a payload is given, GET otherwise. Bearer auth on an absolute URL. */
function bedrockFetch(url: string, payload?: unknown, signal?: AbortSignal): Promise<Response> {
  const token = process.env.AWS_BEARER_TOKEN_BEDROCK;
  if (!token) throw new ConfigError("AWS_BEARER_TOKEN_BEDROCK is required");
  if (payload !== undefined) logBody(`upstream ${url}`, payload);
  return fetch(url, {
    method: payload === undefined ? "GET" : "POST",
    headers: {
      authorization: `Bearer ${token}`,
      ...(payload !== undefined && { "content-type": "application/json" }),
    },
    ...(payload !== undefined && { body: JSON.stringify(payload) }),
    signal,
  });
}

async function fetchJSON(url: string): Promise<unknown> {
  const res = await bedrockFetch(url);
  if (!res.ok) throw new UpstreamError(res.status || 502, await res.text());
  return res.json();
}

const catalogue = makeCatalogue(fetchJSON, CONTROL);

/** Set by the handlers so the log line can name the model. */
type Vars = { model?: string; streaming?: boolean };

const app = new Hono<{ Variables: Vars }>();

// One line per request: method, path, status, duration, upstream model. On streaming
// requests the duration is time-to-handler-return (roughly time to first byte), not the
// length of the stream — the handler returns while SSE is still flowing.
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
    return c.json({ error: { message: "rate limit exceeded", type: "rate_limit_error" } }, 429, {
      "retry-after": "60",
    });
  }
  return next();
});

app.get("/health", (c) => c.json({ ok: true, model: DEFAULT_MODEL, region: REGION }));

app.get("/v1/models", async (c) => {
  const entries = [...(await catalogue.all()).values()];
  return c.json(modelList(entries));
});

app.get("/v1/models/:id", async (c) => {
  const id = c.req.param("id");
  c.set("model", id);
  const found = await catalogue.find(id);
  if (!found) {
    return c.json(
      { type: "error", error: { type: "not_found_error", message: `model "${id}" not found` } },
      404,
    );
  }
  return c.json(modelList([found]).data[0]!);
});

type Route = { shown: string; entry: Entry };

/**
 * Resolve the client's model id against the live catalogue. resolveModel only normalizes
 * the name and picks a same-family fallback; the catalogue decides what is callable and
 * under which id.
 */
async function route(requested: string | undefined, c: Context): Promise<Route> {
  const { model } = resolveModel(requested);
  const shown = requested ?? model;

  let entry = await catalogue.find(model);
  if (!entry && model !== DEFAULT_MODEL) {
    entry = await catalogue.find(DEFAULT_MODEL);
    if (entry) console.warn(`model "${model}" is not in the catalogue — using ${entry.id}`);
  }
  if (!entry) {
    throw new UpstreamError(
      404,
      `model "${model}" is not in this account's catalogue, and DEFAULT_MODEL (${DEFAULT_MODEL}) isn't either. Check GET /v1/models.`,
    );
  }

  c.set("model", entry.upstreamId);
  return { shown, entry };
}

/**
 * A stream that dies after the headers are sent cannot become an HTTP error code, so it
 * has to end with an in-band error event. Without one the client sits waiting for a
 * terminator that never arrives — Anthropic's own API emits `event: error` here, so
 * clients already know the shape.
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

/** Calls Converse (or ConverseStream) with an already-translated payload. */
async function converse(route: Route, payload: unknown, streaming: boolean, signal: AbortSignal) {
  const path = streaming ? "converse-stream" : "converse";
  const res = await bedrockFetch(
    `${RUNTIME}/model/${route.entry.upstreamId}/${path}`,
    payload,
    signal,
  );
  if (!res.ok) throw new UpstreamError(res.status || 502, await res.text());
  if (streaming && !res.body) throw new UpstreamError(502, "converse-stream returned no body");
  return res;
}

/** Anthropic-dialect request and response. */
async function asMessages(
  c: Context,
  p: Anthropic.MessageCreateParams,
  route: Route,
  streaming: boolean,
) {
  const payload = messagesToConverse(
    {
      ...p,
      system: mergeSystem(
        p.system as string | Anthropic.TextBlockParam[] | undefined,
        route.entry.caching && countBreakpoints(p) < 4,
      ),
    },
    route.entry.caching,
  );
  const upstream = await converse(route, payload, streaming, c.req.raw.signal);

  if (!streaming) {
    const body = (await upstream.json()) as ConverseResponse;
    logBody("upstream response", body);
    return c.json(converseToMessage(body, route.shown));
  }

  return streamSSE(c, async (sse) => {
    const emitter = makeConverseEmitter(route.shown);
    try {
      for await (const frame of eventStreamFrames(upstream.body!)) {
        for (const ev of emitter.push(frame.event, frame.payload)) {
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

/**
 * OpenAI-dialect request and response. Goes OpenAI -> Anthropic -> Converse and back, so
 * both legs reuse translators that already exist and are already tested.
 */
async function asChat(c: Context, body: OpenAIRequest, route: Route) {
  const streaming = !!body.stream;
  const asAnthropic = normalize(openAIToAnthropic(body), streaming) as Anthropic.MessageCreateParams;
  const payload = messagesToConverse(asAnthropic, route.entry.caching);
  const upstream = await converse(route, payload, streaming, c.req.raw.signal);

  if (!streaming) {
    const res = (await upstream.json()) as ConverseResponse;
    logBody("upstream response", res);
    return c.json(toChatCompletion(converseToMessage(res, route.shown), route.shown));
  }

  return streamSSE(c, async (sse) => {
    const emitter = makeConverseEmitter(route.shown);
    const toChunks = makeChunker(route.shown);
    try {
      // Converse events become Anthropic events, then OpenAI chunks — one pipeline.
      const emit = async (events: { type: string }[]) => {
        for (const ev of events) {
          for (const chunk of toChunks(ev as Anthropic.MessageStreamEvent)) {
            await sse.writeSSE({ data: JSON.stringify(chunk) });
          }
        }
      };
      for await (const frame of eventStreamFrames(upstream.body!)) {
        await emit(emitter.push(frame.event, frame.payload));
      }
      await emit(emitter.finish());
      await sse.writeSSE({ data: "[DONE]" });
    } catch (err) {
      await sseFailure(sse, "openai", err);
    }
  });
}

app.post("/v1/messages", async (c) => {
  const { stream, ...rest } = await c.req.json<Anthropic.MessageCreateParams>();
  const streaming = !!stream;
  c.set("streaming", streaming);
  const target = await route(rest.model, c);
  return asMessages(c, rest as Anthropic.MessageCreateParams, target, streaming);
});

app.post("/v1/chat/completions", async (c) => {
  const body = await c.req.json<OpenAIRequest>();
  c.set("streaming", !!body.stream);
  const target = await route(body.model, c);
  return asChat(c, body, target);
});

// Bedrock Runtime counts tokens against the Converse shape, so the same translation
// feeds it. Needs the bedrock:CountTokens IAM action — see the README.
app.post("/v1/messages/count_tokens", async (c) => {
  const body = await c.req.json<Anthropic.MessageCountTokensParams>();
  const target = await route(body.model, c);

  const converseBody = messagesToConverse(
    {
      ...body,
      max_tokens: 1, // unused by count-tokens, but the translator expects a request shape
      system: mergeSystem(
        body.system as string | Anthropic.TextBlockParam[] | undefined,
        target.entry.caching && countBreakpoints(body) < 4,
      ),
    } as Anthropic.MessageCreateParams,
    target.entry.caching,
  );
  delete converseBody.inferenceConfig;

  const res = await bedrockFetch(
    `${RUNTIME}/model/${target.entry.upstreamId}/count-tokens`,
    { input: { converse: converseBody } },
    c.req.raw.signal,
  );
  if (!res.ok) throw new UpstreamError(res.status || 502, await res.text());
  const counted = (await res.json()) as { inputTokens?: number };
  return c.json({ input_tokens: counted.inputTokens ?? 0 });
});

// Converse has no Responses API. Say so plainly instead of 404ing.
app.post("/v1/responses", (c) =>
  c.json(
    {
      error: {
        message:
          "Bedrock Converse has no Responses API. Point the client at /v1/chat/completions instead — in Codex that is wire_api = \"chat\".",
        type: "invalid_request_error",
      },
    },
    400,
  ),
);

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
console.log(`zyroute on http://${hostname}:${port} → bedrock ${REGION || "(no region)"}`);

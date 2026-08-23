import { readFileSync } from "node:fs";
import path from "node:path";
import type Anthropic from "@anthropic-ai/sdk";

/**
 * Edit ../system-prompt.md, not this file. Read once at startup: the prompt has to
 * be byte-stable for prompt caching to hit, and re-reading per request would also
 * mean a mid-flight edit could half-apply. Restart to pick up changes.
 *
 * A missing or empty file throws here rather than silently serving an unguarded
 * model.
 */
const PROMPT_FILE =
  process.env.SYSTEM_PROMPT_FILE ?? path.join(import.meta.dirname, "../system-prompt.md");

export const BASE_SYSTEM = readFileSync(PROMPT_FILE, "utf8").trim();
if (!BASE_SYSTEM) throw new Error(`base system prompt is empty: ${PROMPT_FILE}`);

export const DEFAULT_MODEL = process.env.DEFAULT_MODEL ?? "anthropic.claude-opus-4-7";
export const DEFAULT_OPENAI_MODEL = process.env.DEFAULT_OPENAI_MODEL ?? "openai.gpt-oss-120b-1:0";

/** Internal request shape: Anthropic Messages params without `stream`. */
export type Params = Omit<Anthropic.MessageCreateParamsNonStreaming, "stream">;

/**
 * Models that reject temperature/top_p/top_k with a 400. This list is the one part of
 * the gateway that rots as Anthropic ships models, so it is overridable: set
 * NO_SAMPLING_MODELS to a regex if a newer model starts rejecting sampling too.
 */
const NO_SAMPLING = new RegExp(
  process.env.NO_SAMPLING_MODELS ?? "(fable-5|mythos-5|opus-5|opus-4-8|opus-4-7|sonnet-5)",
);

/**
 * Base prompt first, client prompt appended after it.
 *
 * `cacheBase` exists because Anthropic allows at most 4 cache_control breakpoints per
 * request and Claude Code routinely uses all four — adding a fifth 400s the request.
 * See countBreakpoints().
 */
export function mergeSystem(
  requested?: string | Anthropic.TextBlockParam[] | null,
  cacheBase = true,
): Anthropic.TextBlockParam[] {
  const base: Anthropic.TextBlockParam = {
    type: "text",
    text: BASE_SYSTEM,
    ...(cacheBase && { cache_control: { type: "ephemeral" as const } }), // stable prefix
  };
  if (!requested) return [base];
  const extra: Anthropic.TextBlockParam[] =
    typeof requested === "string"
      ? requested.trim()
        ? [{ type: "text", text: requested }]
        : []
      : requested;
  return [base, ...extra];
}

const warned = new Set<string>();

/**
 * Bedrock namespaces every model as `provider.model` — anthropic.claude-opus-4-7,
 * amazon.nova-pro-v1:0, meta.llama3-3-70b-instruct-v1:0. The prefix before the first dot
 * is a bare provider name, which is what separates a real Bedrock id from a foreign one:
 * "gpt-4.1" has a dot but its prefix ("gpt-4") has a dash, so it is an OpenAI-native id,
 * not a Bedrock one.
 */
const BEDROCK_ID = /^[a-z][a-z0-9]*\.\S+$/;

/** Only used to keep a fallback inside the family the client asked for. */
export type Family = "anthropic" | "openai";

/** Foreign ids that plainly name an OpenAI-family model rather than a Bedrock one. */
const LOOKS_OPENAI = /^(gpt|chatgpt|codex|davinci|o[1-9])/i;

/**
 * Whatever model the client selected wins. Only a genuinely foreign id — "gpt-4o"
 * from an OpenAI client — falls back, and says so once, because a silently swapped
 * model is miserable to debug.
 *
 * The fallback stays inside the family the client asked for: a request for "gpt-4o"
 * lands on DEFAULT_OPENAI_MODEL, not on Claude. Swapping families changes tokenizer,
 * tool semantics and price all at once, which is not a substitution anyone wants made
 * on their behalf.
 */
export function resolveModel(model?: string): { model: string; family: Family } {
  const tag = (m: string) => ({ model: m, family: familyOf(m) });
  if (!model) return tag(DEFAULT_MODEL);
  if (model.startsWith("claude")) return tag(`anthropic.${model}`); // any Claude id, dated or not
  if (BEDROCK_ID.test(model)) return tag(model); // incl. us./eu./apac. inference profiles

  const fallback = LOOKS_OPENAI.test(model) ? DEFAULT_OPENAI_MODEL : DEFAULT_MODEL;
  if (!warned.has(model)) {
    // the model id comes from the client, so this set is capped rather than left to
    // grow on a stream of junk ids
    if (warned.size > 100) warned.clear();
    warned.add(model);
    console.warn(`model "${model}" is not a Bedrock model id — using ${fallback} instead`);
  }
  return tag(fallback);
}

function familyOf(model: string): Family {
  return model.includes("anthropic.") ? "anthropic" : "openai";
}

/** Base prompt as the first system message, client's messages untouched behind it. */
export function withBaseSystem(msgs: OpenAIMessage[]): OpenAIMessage[] {
  return [{ role: "system", content: BASE_SYSTEM }, ...msgs];
}

/**
 * One model-list payload that satisfies both dialects: OpenAI clients read
 * `object`/`created`/`owned_by`, Anthropic clients read `type`/`display_name`/
 * `created_at`. Each ignores the other's extra fields, so no content negotiation is
 * needed. Everything comes from the upstream list — nothing is invented here.
 */
export function modelList(models: { id: string; created?: number; owned_by?: string }[]) {
  const data = models.map((m) => ({
    id: m.id,
    object: "model",
    type: "model",
    created: m.created ?? 0,
    created_at: new Date((m.created ?? 0) * 1000).toISOString(),
    display_name: m.id,
    owned_by: m.owned_by ?? "bedrock",
  }));
  return {
    object: "list",
    data,
    has_more: false,
    first_id: data[0]?.id ?? null,
    last_id: data.at(-1)?.id ?? null,
  };
}

/** Anthropic caps cache_control breakpoints at 4 per request; ours has to fit. */
export function countBreakpoints(value: unknown): number {
  if (Array.isArray(value)) return value.reduce<number>((n, v) => n + countBreakpoints(v), 0);
  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return (
      (obj.cache_control ? 1 : 0) +
      Object.values(obj).reduce<number>((n, v) => n + countBreakpoints(v), 0)
    );
  }
  return 0;
}

/** Applied to every call, whichever dialect it arrived in. */
export function normalize(p: Params, streaming = false): Params {
  const { model } = resolveModel(p.model);
  const out: Params = {
    ...p,
    model,
    system: mergeSystem(
      p.system as string | Anthropic.TextBlockParam[] | undefined,
      countBreakpoints(p) < 4,
    ),
    max_tokens: p.max_tokens ?? (streaming ? 64000 : 16000),
  };
  if (NO_SAMPLING.test(model)) {
    delete out.temperature;
    delete out.top_p;
    delete out.top_k;
  }
  return out;
}

// ─── OpenAI dialect ────────────────────────────────────────────────────────────

type Part = { type: string; text?: string; image_url?: { url: string } };

export type OpenAIMessage = {
  role: "system" | "developer" | "user" | "assistant" | "tool";
  content?: string | Part[] | null;
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
  tool_call_id?: string;
};

export type OpenAIRequest = {
  model?: string;
  messages?: OpenAIMessage[];
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  stream?: boolean;
  reasoning_effort?: "minimal" | "low" | "medium" | "high";
  tools?: { type: "function"; function: { name: string; description?: string; parameters?: Record<string, unknown> } }[];
  tool_choice?: "auto" | "none" | "required" | { type: "function"; function: { name: string } };
  response_format?: { type: string; json_schema?: { schema: Record<string, unknown> } };
};

const FINISH: Record<string, string> = {
  end_turn: "stop",
  stop_sequence: "stop",
  max_tokens: "length",
  tool_use: "tool_calls",
  pause_turn: "stop",
  refusal: "content_filter",
};

const EFFORT = { minimal: "low", low: "low", medium: "medium", high: "high" } as const;

function textOf(content: OpenAIMessage["content"]): string {
  if (typeof content === "string") return content;
  if (!content) return "";
  return content.filter((p) => p.type === "text").map((p) => p.text ?? "").join("");
}

function imageBlock(url: string): Anthropic.ImageBlockParam {
  const m = /^data:([^;]+);base64,(.*)$/s.exec(url);
  return m
    ? { type: "image", source: { type: "base64", media_type: m[1] as "image/png", data: m[2]! } }
    : { type: "image", source: { type: "url", url } };
}

function userContent(content: OpenAIMessage["content"]): string | Anthropic.ContentBlockParam[] {
  if (typeof content === "string" || content == null) return content ?? "";
  return content.map((p) =>
    p.type === "image_url" && p.image_url
      ? imageBlock(p.image_url.url)
      : ({ type: "text", text: p.text ?? "" } as Anthropic.TextBlockParam),
  );
}

export function parseArgs(json: string): Record<string, unknown> {
  try {
    return JSON.parse(json || "{}");
  } catch {
    return {};
  }
}

/** OpenAI messages[] → Anthropic system + messages. */
export function splitMessages(msgs: OpenAIMessage[]): {
  system: string;
  messages: Anthropic.MessageParam[];
} {
  const system: string[] = [];
  const messages: Anthropic.MessageParam[] = [];

  for (const m of msgs) {
    if (m.role === "system" || m.role === "developer") {
      const t = textOf(m.content);
      if (t) system.push(t);
      continue;
    }

    if (m.role === "tool") {
      const block: Anthropic.ToolResultBlockParam = {
        type: "tool_result",
        tool_use_id: m.tool_call_id ?? "",
        content: textOf(m.content),
      };
      // Every tool_result answering one assistant turn must ride in a single
      // user message, or Claude stops emitting parallel tool calls.
      const last = messages[messages.length - 1];
      const isResultBag =
        last?.role === "user" &&
        Array.isArray(last.content) &&
        last.content.every((b) => b.type === "tool_result");
      if (isResultBag) (last!.content as Anthropic.ContentBlockParam[]).push(block);
      else messages.push({ role: "user", content: [block] });
      continue;
    }

    if (m.role === "assistant") {
      const content: Anthropic.ContentBlockParam[] = [];
      const t = textOf(m.content);
      if (t) content.push({ type: "text", text: t });
      for (const c of m.tool_calls ?? [])
        content.push({
          type: "tool_use",
          id: c.id,
          name: c.function.name,
          input: parseArgs(c.function.arguments),
        });
      if (content.length) messages.push({ role: "assistant", content });
      continue;
    }

    messages.push({ role: "user", content: userContent(m.content) });
  }

  return { system: system.join("\n\n"), messages };
}

export function openAIToAnthropic(body: OpenAIRequest): Params {
  const { system, messages } = splitMessages(body.messages ?? []);
  const schema = body.response_format?.json_schema?.schema;
  const effort = body.reasoning_effort ? EFFORT[body.reasoning_effort] : undefined;

  return {
    model: body.model ?? "",
    max_tokens: body.max_completion_tokens ?? body.max_tokens ?? (body.stream ? 64000 : 16000),
    messages,
    ...(system && { system }),
    ...(body.temperature != null && { temperature: body.temperature }),
    ...(body.top_p != null && { top_p: body.top_p }),
    ...(body.stop && { stop_sequences: Array.isArray(body.stop) ? body.stop : [body.stop] }),
    ...(body.tools?.length && {
      tools: body.tools.map((t) => ({
        name: t.function.name,
        ...(t.function.description && { description: t.function.description }),
        input_schema: (t.function.parameters ?? {
          type: "object",
          properties: {},
        }) as Anthropic.Tool.InputSchema,
      })),
    }),
    ...(body.tool_choice && { tool_choice: toToolChoice(body.tool_choice) }),
    ...((effort || schema) && {
      output_config: {
        ...(effort && { effort }),
        ...(schema && { format: { type: "json_schema" as const, schema } }),
      },
    }),
  };
}

function toToolChoice(tc: NonNullable<OpenAIRequest["tool_choice"]>): Anthropic.ToolChoice {
  if (tc === "auto") return { type: "auto" };
  if (tc === "none") return { type: "none" };
  if (tc === "required") return { type: "any" };
  return { type: "tool", name: tc.function.name };
}

export function toChatCompletion(msg: Anthropic.Message, model: string) {
  const text = msg.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
  const toolCalls = msg.content
    .filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use")
    .map((b) => ({
      id: b.id,
      type: "function" as const,
      function: { name: b.name, arguments: JSON.stringify(b.input) },
    }));

  return {
    id: msg.id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: text || null,
          ...(toolCalls.length && { tool_calls: toolCalls }),
        },
        finish_reason: FINISH[msg.stop_reason ?? "end_turn"] ?? "stop",
      },
    ],
    usage: {
      prompt_tokens: msg.usage.input_tokens,
      completion_tokens: msg.usage.output_tokens,
      total_tokens: msg.usage.input_tokens + msg.usage.output_tokens,
    },
  };
}

/**
 * Per-request Anthropic stream event → OpenAI chunk(s). Stateful: tool_use
 * argument deltas carry no id, so the current tool index has to be remembered.
 */
export function makeChunker(model: string) {
  const created = Math.floor(Date.now() / 1000);
  let id = "chatcmpl-pending";
  let promptTokens = 0;
  let toolIndex = -1;

  const chunk = (delta: object, finish: string | null = null, usage?: object) => ({
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
    ...(usage && { usage }),
  });

  return (ev: Anthropic.MessageStreamEvent): object[] => {
    switch (ev.type) {
      case "message_start":
        id = ev.message.id;
        promptTokens = ev.message.usage.input_tokens;
        return [chunk({ role: "assistant", content: "" })];

      case "content_block_start":
        if (ev.content_block.type !== "tool_use") return [];
        toolIndex++;
        return [
          chunk({
            tool_calls: [
              {
                index: toolIndex,
                id: ev.content_block.id,
                type: "function",
                function: { name: ev.content_block.name, arguments: "" },
              },
            ],
          }),
        ];

      case "content_block_delta":
        if (ev.delta.type === "text_delta") return [chunk({ content: ev.delta.text })];
        if (ev.delta.type === "thinking_delta")
          return [chunk({ reasoning_content: ev.delta.thinking })];
        if (ev.delta.type === "input_json_delta")
          return [
            chunk({
              tool_calls: [{ index: toolIndex, function: { arguments: ev.delta.partial_json } }],
            }),
          ];
        return [];

      case "message_delta":
        return [
          chunk({}, FINISH[ev.delta.stop_reason ?? "end_turn"] ?? "stop", {
            prompt_tokens: promptTokens,
            completion_tokens: ev.usage.output_tokens,
            total_tokens: promptTokens + ev.usage.output_tokens,
          }),
        ];

      default:
        return [];
    }
  };
}

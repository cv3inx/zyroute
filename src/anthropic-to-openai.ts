/**
 * The reverse bridge: an Anthropic-dialect client (Claude Code) driving a model that
 * Mantle only serves on its OpenAI surface (openai.gpt-oss-120b, qwen.*, google.*).
 *
 * translate.ts goes OpenAI -> Anthropic. This file goes Anthropic -> OpenAI for the
 * request, and OpenAI -> Anthropic for the response and the SSE stream.
 */
import type Anthropic from "@anthropic-ai/sdk";
import { parseArgs, type OpenAIMessage } from "./translate.ts";

type Part = { type: string; text?: string; image_url?: { url: string } };

export type ChatRequest = {
  model: string;
  messages: OpenAIMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
  stream?: boolean;
  stream_options?: { include_usage: boolean };
  reasoning_effort?: string;
  tools?: { type: "function"; function: { name: string; description?: string; parameters: Record<string, unknown> } }[];
  tool_choice?: "auto" | "none" | "required" | { type: "function"; function: { name: string } };
};

/** OpenAI finish_reason -> Anthropic stop_reason. */
const STOP: Record<string, string> = {
  stop: "end_turn",
  length: "max_tokens",
  tool_calls: "tool_use",
  function_call: "tool_use",
  content_filter: "refusal",
};

/** Anthropic effort levels are wider than OpenAI's. */
const EFFORT: Record<string, string> = {
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "high",
  max: "high",
};

function systemText(system?: string | Anthropic.TextBlockParam[] | null): string {
  if (!system) return "";
  if (typeof system === "string") return system;
  return system
    .map((b) => b.text)
    .filter(Boolean)
    .join("\n\n");
}

function imageURL(b: Anthropic.ImageBlockParam): string {
  const s = b.source;
  if (s.type === "base64") return `data:${s.media_type};base64,${s.data}`;
  if (s.type === "url") return s.url;
  return ""; // file_id sources have no URL form; upstream will reject it loudly
}

function toolResultText(b: Anthropic.ToolResultBlockParam): string {
  if (typeof b.content === "string") return b.content;
  return (b.content ?? [])
    .map((part) => (part.type === "text" ? part.text : ""))
    .filter(Boolean)
    .join("\n");
}

/**
 * Anthropic Messages request -> OpenAI Chat Completions request.
 *
 * `streaming` is an explicit argument, not read off `p.stream`: callers routinely
 * destructure `stream` out of the incoming body before getting here, and inferring it
 * silently sent a non-streaming request upstream while the client waited on SSE.
 */
export function messagesToChat(
  p: Anthropic.MessageCreateParams,
  model: string,
  streaming = false,
): ChatRequest {
  const messages: OpenAIMessage[] = [];

  const sys = systemText(p.system as string | Anthropic.TextBlockParam[] | undefined);
  if (sys) messages.push({ role: "system", content: sys });

  for (const m of p.messages) {
    if (typeof m.content === "string") {
      messages.push({ role: m.role, content: m.content });
      continue;
    }

    if (m.role === "assistant") {
      let text = "";
      const toolCalls: NonNullable<OpenAIMessage["tool_calls"]> = [];
      for (const b of m.content) {
        if (b.type === "text") text += b.text;
        else if (b.type === "tool_use") {
          toolCalls.push({
            id: b.id,
            type: "function",
            function: { name: b.name, arguments: JSON.stringify(b.input ?? {}) },
          });
        }
        // thinking / redacted_thinking blocks are dropped: gpt-oss reasoning is not
        // interchangeable with Anthropic's, and replaying it would be rejected.
      }
      messages.push({
        role: "assistant",
        content: text || null,
        ...(toolCalls.length && { tool_calls: toolCalls }),
      });
      continue;
    }

    // A user turn carries tool_results, text and images. Each tool_result becomes its
    // own `tool` message; the rest collapses into one user message.
    const parts: Part[] = [];
    for (const b of m.content) {
      if (b.type === "tool_result") {
        messages.push({ role: "tool", tool_call_id: b.tool_use_id, content: toolResultText(b) });
      } else if (b.type === "text") {
        parts.push({ type: "text", text: b.text });
      } else if (b.type === "image") {
        parts.push({ type: "image_url", image_url: { url: imageURL(b) } });
      }
    }
    if (parts.length) messages.push({ role: "user", content: parts });
  }

  const tools = (p.tools ?? [])
    .filter((t): t is Anthropic.Tool => "input_schema" in t) // server tools have no schema
    .map((t) => ({
      type: "function" as const,
      function: {
        name: t.name,
        ...(t.description && { description: t.description }),
        parameters: t.input_schema as unknown as Record<string, unknown>,
      },
    }));

  const effort = p.output_config?.effort ? EFFORT[p.output_config.effort] : undefined;

  return {
    model,
    messages,
    max_tokens: p.max_tokens,
    ...(p.temperature != null && { temperature: p.temperature }),
    ...(p.top_p != null && { top_p: p.top_p }),
    ...(p.stop_sequences?.length && { stop: p.stop_sequences }),
    ...(tools.length && { tools }),
    ...(p.tool_choice && { tool_choice: toolChoice(p.tool_choice) }),
    ...(effort && { reasoning_effort: effort }),
    ...(streaming && { stream: true, stream_options: { include_usage: true } }),
  };
}

function toolChoice(tc: Anthropic.ToolChoice): ChatRequest["tool_choice"] {
  switch (tc.type) {
    case "auto":
      return "auto";
    case "any":
      return "required";
    case "none":
      return "none";
    case "tool":
      return { type: "function", function: { name: tc.name } };
  }
}

type ChatChoice = {
  message?: { content?: string | null; tool_calls?: { id: string; function: { name: string; arguments: string } }[] };
  finish_reason?: string | null;
};
export type ChatResponse = {
  id?: string;
  choices?: ChatChoice[];
  usage?: { prompt_tokens?: number; completion_tokens?: number };
};

/** OpenAI Chat Completion -> Anthropic Message. */
export function chatToMessage(cc: ChatResponse, model: string): Anthropic.Message {
  const choice = cc.choices?.[0] ?? {};
  const content: Anthropic.ContentBlock[] = [];

  if (choice.message?.content) {
    content.push({ type: "text", text: choice.message.content, citations: null });
  }
  for (const tc of choice.message?.tool_calls ?? []) {
    content.push({
      type: "tool_use",
      id: tc.id,
      name: tc.function.name,
      input: parseArgs(tc.function.arguments),
      caller: { type: "direct" },
    });
  }
  // Anthropic clients expect at least one block.
  if (!content.length) content.push({ type: "text", text: "", citations: null });

  return {
    id: `msg_${cc.id ?? "bridge"}`,
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: (STOP[choice.finish_reason ?? "stop"] ?? "end_turn") as Anthropic.StopReason,
    stop_sequence: null,
    usage: {
      input_tokens: cc.usage?.prompt_tokens ?? 0,
      output_tokens: cc.usage?.completion_tokens ?? 0,
    },
  } as Anthropic.Message;
}

type ChatChunk = {
  id?: string;
  usage?: { prompt_tokens?: number; completion_tokens?: number } | null;
  choices?: {
    delta?: {
      content?: string | null;
      tool_calls?: { index: number; id?: string; function?: { name?: string; arguments?: string } }[];
    };
    finish_reason?: string | null;
  }[];
};

type Event = { type: string } & Record<string, unknown>;

/**
 * OpenAI stream chunks -> Anthropic SSE events.
 *
 * Anthropic's stream has exactly one content block open at a time, so a new block
 * closes the previous one. OpenAI servers stream tool calls one after another, which
 * fits that shape.
 *
 * ponytail: if an upstream ever interleaves deltas for two tool indices, the second
 * one's arguments would land after its block closed. Buffer per index and flush at
 * the end if you hit a backend that does that.
 */
export function makeMessageEmitter(model: string) {
  let started = false;
  let openIndex: number | null = null;
  let nextIndex = 0;
  let textIndex: number | null = null;
  const toolIndexes = new Map<number, number>();
  let stopReason = "end_turn";
  let inputTokens = 0;
  let outputTokens = 0;

  const closeOpen = (out: Event[]) => {
    if (openIndex !== null) {
      out.push({ type: "content_block_stop", index: openIndex });
      openIndex = null;
    }
  };

  const start = (id: string | undefined, out: Event[]) => {
    if (started) return;
    started = true;
    out.push({
      type: "message_start",
      message: {
        id: `msg_${id ?? "bridge"}`,
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    });
  };

  return {
    push(chunk: ChatChunk): Event[] {
      const out: Event[] = [];
      if (chunk.usage) {
        inputTokens = chunk.usage.prompt_tokens ?? inputTokens;
        outputTokens = chunk.usage.completion_tokens ?? outputTokens;
      }
      start(chunk.id, out);

      const choice = chunk.choices?.[0];
      if (!choice) return out;
      const delta = choice.delta ?? {};

      if (delta.content) {
        if (textIndex === null || openIndex !== textIndex) {
          closeOpen(out);
          textIndex = nextIndex++;
          openIndex = textIndex;
          out.push({
            type: "content_block_start",
            index: textIndex,
            content_block: { type: "text", text: "" },
          });
        }
        out.push({
          type: "content_block_delta",
          index: textIndex,
          delta: { type: "text_delta", text: delta.content },
        });
      }

      for (const tc of delta.tool_calls ?? []) {
        let index = toolIndexes.get(tc.index);
        if (index === undefined) {
          closeOpen(out);
          index = nextIndex++;
          toolIndexes.set(tc.index, index);
          openIndex = index;
          out.push({
            type: "content_block_start",
            index,
            content_block: {
              type: "tool_use",
              id: tc.id ?? `toolu_bridge_${index}`,
              name: tc.function?.name ?? "",
              input: {},
            },
          });
        }
        if (tc.function?.arguments) {
          out.push({
            type: "content_block_delta",
            index,
            delta: { type: "input_json_delta", partial_json: tc.function.arguments },
          });
        }
      }

      if (choice.finish_reason) stopReason = STOP[choice.finish_reason] ?? "end_turn";
      return out;
    },

    finish(): Event[] {
      const out: Event[] = [];
      start(undefined, out);
      closeOpen(out);
      out.push({
        type: "message_delta",
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens },
      });
      out.push({ type: "message_stop" });
      return out;
    },
  };
}

/** Yields parsed `data:` payloads from an SSE body, stopping at [DONE]. */
export async function* sseJSON(body: ReadableStream<Uint8Array>): AsyncGenerator<ChatChunk> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffered += decoder.decode(value, { stream: true });
    const lines = buffered.split("\n");
    buffered = lines.pop() ?? ""; // keep the trailing partial line

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      if (payload === "[DONE]") return;
      try {
        yield JSON.parse(payload) as ChatChunk;
      } catch {
        // a malformed chunk is not worth killing the stream over
      }
    }
  }
}

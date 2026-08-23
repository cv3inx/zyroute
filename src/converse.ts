/**
 * The only upstream: the Bedrock Runtime **Converse** API.
 *
 * Bedrock Mantle was the first target here and has been dropped. It carried 55 models
 * against the account's 121 foundation models, had no Claude Opus 4.6 at all, and needed
 * three separate upstream surfaces to cover. Converse reaches everything — every Claude,
 * Nova, Llama, Mistral, Cohere, AI21, Qwen — over one protocol with one bearer token.
 *
 * One translator pair lives here, Anthropic Messages <-> Converse. The OpenAI dialect
 * reaches Converse through the existing OpenAI->Anthropic translation first, so there is
 * no second implementation to keep in step.
 */
import type Anthropic from "@anthropic-ai/sdk";

type ConverseBlock = Record<string, unknown>;

export type ConverseRequest = {
  messages: { role: "user" | "assistant"; content: ConverseBlock[] }[];
  system?: ConverseBlock[];
  additionalModelRequestFields?: Record<string, unknown>;
  inferenceConfig?: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };
  toolConfig?: {
    tools: { toolSpec: { name: string; description?: string; inputSchema: { json: unknown } } }[];
    toolChoice?: Record<string, unknown>;
  };
};

/** Converse stopReason -> Anthropic stop_reason. */
const STOP: Record<string, string> = {
  end_turn: "end_turn",
  max_tokens: "max_tokens",
  stop_sequence: "stop_sequence",
  tool_use: "tool_use",
  content_filtered: "refusal",
  guardrail_intervened: "refusal",
};

/** Converse wants a bare format name plus base64 bytes, not a data URI. */
function imageBlock(b: Anthropic.ImageBlockParam): ConverseBlock | null {
  if (b.source.type !== "base64") return null; // url/file sources aren't supported here
  const format = b.source.media_type.replace("image/", "").replace("jpg", "jpeg");
  return { image: { format, source: { bytes: b.source.data } } };
}

function toolResultBlock(b: Anthropic.ToolResultBlockParam): ConverseBlock {
  const text =
    typeof b.content === "string"
      ? b.content
      : (b.content ?? [])
          .map((part) => (part.type === "text" ? part.text : ""))
          .filter(Boolean)
          .join("\n");
  return {
    toolResult: {
      toolUseId: b.tool_use_id,
      content: [{ text: text || "(no output)" }],
      ...(b.is_error && { status: "error" }),
    },
  };
}

/**
 * `caching` is passed in from the catalogue, not inferred here: Claude and Nova accept
 * cachePoint blocks while Llama and gpt-oss reject the entire request, so a client's
 * cache_control marker can only be honoured on models that support it.
 */
export function messagesToConverse(
  p: Anthropic.MessageCreateParams,
  caching = false,
): ConverseRequest {
  const messages: ConverseRequest["messages"] = [];
  const operator: string[] = [];
  const cachePoint = () => ({ cachePoint: { type: "default" } });

  for (const m of p.messages) {
    // Converse takes user/assistant only. A mid-conversation system message is an
    // operator instruction, so it joins the system blocks rather than being demoted to
    // a user turn, which would hand it the wrong authority.
    if (m.role === "system") {
      const text =
        typeof m.content === "string"
          ? m.content
          : m.content
              .map((b) => (b.type === "text" ? b.text : ""))
              .filter(Boolean)
              .join("\n");
      if (text) operator.push(text);
      continue;
    }

    const content: ConverseBlock[] = [];

    if (typeof m.content === "string") {
      if (m.content) content.push({ text: m.content });
    } else {
      for (const b of m.content) {
        if (b.type === "text") {
          if (b.text) content.push({ text: b.text });
          if (caching && b.cache_control) content.push(cachePoint());
        } else if (b.type === "image") {
          const image = imageBlock(b);
          if (image) content.push(image);
        } else if (b.type === "tool_use") {
          content.push({ toolUse: { toolUseId: b.id, name: b.name, input: b.input ?? {} } });
        } else if (b.type === "tool_result") {
          content.push(toolResultBlock(b));
        }
        // thinking blocks are dropped: Converse has its own reasoningContent shape and
        // the signatures are not interchangeable.
      }
    }

    // Converse rejects a message with an empty content array.
    if (content.length) messages.push({ role: m.role, content });
  }

  const tools = (p.tools ?? [])
    .filter((t): t is Anthropic.Tool => "input_schema" in t)
    .map((t) => ({
      toolSpec: {
        name: t.name,
        ...(t.description && { description: t.description }),
        inputSchema: { json: t.input_schema },
      },
    }));

  const system: ConverseBlock[] = [];
  const requested = p.system;
  if (Array.isArray(requested)) {
    for (const b of requested as Anthropic.TextBlockParam[]) {
      if (b.text?.trim()) system.push({ text: b.text });
      if (caching && b.cache_control) system.push(cachePoint());
    }
  } else if (typeof requested === "string" && requested.trim()) {
    system.push({ text: requested });
  }
  for (const text of operator) system.push({ text });

  return {
    messages,
    ...(system.length && { system }),
    // Extended thinking and any other model-specific knob ride through untouched.
    ...(p.thinking && { additionalModelRequestFields: { thinking: p.thinking } }),
    inferenceConfig: {
      ...(p.max_tokens && { maxTokens: p.max_tokens }),
      ...(p.temperature != null && { temperature: p.temperature }),
      ...(p.top_p != null && { topP: p.top_p }),
      ...(p.stop_sequences?.length && { stopSequences: p.stop_sequences }),
    },
    ...(tools.length && {
      toolConfig: { tools, ...(p.tool_choice && { toolChoice: toolChoice(p.tool_choice) }) },
    }),
  };
}

function toolChoice(tc: Anthropic.ToolChoice): Record<string, unknown> | undefined {
  switch (tc.type) {
    case "auto":
      return { auto: {} };
    case "any":
      return { any: {} };
    case "tool":
      return { tool: { name: tc.name } };
    case "none":
      return undefined; // Converse has no "none"; omitting toolChoice is the closest thing
  }
}

export type ConverseResponse = {
  output?: { message?: { content?: ConverseBlock[] } };
  stopReason?: string;
  usage?: { inputTokens?: number; outputTokens?: number };
};

export function converseToMessage(res: ConverseResponse, model: string): Anthropic.Message {
  const content: Anthropic.ContentBlock[] = [];

  for (const block of res.output?.message?.content ?? []) {
    if (typeof block.text === "string" && block.text) {
      content.push({ type: "text", text: block.text, citations: null });
    } else if (block.toolUse) {
      const use = block.toolUse as { toolUseId: string; name: string; input?: unknown };
      content.push({
        type: "tool_use",
        id: use.toolUseId,
        name: use.name,
        input: (use.input ?? {}) as Record<string, unknown>,
        caller: { type: "direct" },
      });
    }
    // reasoningContent is dropped for the same reason as thinking blocks
  }
  if (!content.length) content.push({ type: "text", text: "", citations: null });

  return {
    id: `msg_converse_${Math.random().toString(36).slice(2, 10)}`,
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: (STOP[res.stopReason ?? "end_turn"] ?? "end_turn") as Anthropic.StopReason,
    stop_sequence: null,
    usage: {
      input_tokens: res.usage?.inputTokens ?? 0,
      output_tokens: res.usage?.outputTokens ?? 0,
    },
  } as Anthropic.Message;
}

type Event = { type: string } & Record<string, unknown>;

/**
 * ConverseStream events -> Anthropic SSE events.
 *
 * The two protocols line up closely, with one gap: Converse emits contentBlockDelta for
 * text without any contentBlockStart, while Anthropic clients require the start event.
 * So text blocks are opened lazily on their first delta.
 */
export function makeConverseEmitter(model: string) {
  const open = new Set<number>();
  let stopReason = "end_turn";
  let inputTokens = 0;
  let outputTokens = 0;
  let started = false;

  const openBlock = (index: number, block: Record<string, unknown>, out: Event[]) => {
    if (open.has(index)) return;
    open.add(index);
    out.push({ type: "content_block_start", index, content_block: block });
  };

  return {
    push(event: string, payload: Record<string, unknown>): Event[] {
      const out: Event[] = [];
      const index = (payload.contentBlockIndex as number) ?? 0;

      switch (event) {
        case "messageStart":
          started = true;
          out.push({
            type: "message_start",
            message: {
              id: `msg_converse_${Math.random().toString(36).slice(2, 10)}`,
              type: "message",
              role: "assistant",
              model,
              content: [],
              stop_reason: null,
              stop_sequence: null,
              usage: { input_tokens: 0, output_tokens: 0 },
            },
          });
          break;

        case "contentBlockStart": {
          const start = payload.start as { toolUse?: { toolUseId: string; name: string } };
          if (start?.toolUse) {
            openBlock(
              index,
              {
                type: "tool_use",
                id: start.toolUse.toolUseId,
                name: start.toolUse.name,
                input: {},
              },
              out,
            );
          }
          break;
        }

        case "contentBlockDelta": {
          const delta = payload.delta as {
            text?: string;
            toolUse?: { input?: string };
            reasoningContent?: unknown;
          };
          if (typeof delta?.text === "string") {
            openBlock(index, { type: "text", text: "" }, out);
            out.push({
              type: "content_block_delta",
              index,
              delta: { type: "text_delta", text: delta.text },
            });
          } else if (delta?.toolUse?.input !== undefined) {
            out.push({
              type: "content_block_delta",
              index,
              delta: { type: "input_json_delta", partial_json: delta.toolUse.input },
            });
          }
          break;
        }

        case "contentBlockStop":
          if (open.delete(index)) out.push({ type: "content_block_stop", index });
          break;

        case "messageStop":
          stopReason = STOP[(payload.stopReason as string) ?? "end_turn"] ?? "end_turn";
          break;

        case "metadata": {
          const usage = payload.usage as { inputTokens?: number; outputTokens?: number };
          inputTokens = usage?.inputTokens ?? inputTokens;
          outputTokens = usage?.outputTokens ?? outputTokens;
          break;
        }
      }
      return out;
    },

    finish(): Event[] {
      const out: Event[] = [];
      if (!started) {
        out.push(...this.push("messageStart", {}));
      }
      for (const index of [...open]) {
        open.delete(index);
        out.push({ type: "content_block_stop", index });
      }
      out.push({
        type: "message_delta",
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { input_tokens: inputTokens, output_tokens: outputTokens },
      });
      out.push({ type: "message_stop" });
      return out;
    },
  };
}

import assert from "node:assert/strict";
import type Anthropic from "@anthropic-ai/sdk";
import { makeKeyGuard, makeRateLimiter, presentedKey } from "./src/auth.ts";
import { redact } from "./src/log.ts";
import {
  converseToMessage,
  makeConverseEmitter,
  messagesToConverse,
} from "./src/converse.ts";
import { buildCatalogue } from "./src/catalogue.ts";
import { eventStreamFrames } from "./src/eventstream.ts";
import {
  BASE_SYSTEM,
  countBreakpoints,
  makeChunker,
  modelList,
  mergeSystem,
  normalize,
  openAIToAnthropic,
  splitMessages,
  resolveModel,
  toChatCompletion,
  withBaseSystem,
  type Params,
} from "./src/translate.ts";

// system prompt is appended, never replaced
{
  assert.ok(BASE_SYSTEM.length > 0, "system-prompt.md must not be empty");
  assert.deepEqual(mergeSystem().map((b) => b.text), [BASE_SYSTEM]);
  assert.deepEqual(mergeSystem("be terse").map((b) => b.text), [BASE_SYSTEM, "be terse"]);
  assert.deepEqual(
    mergeSystem([{ type: "text", text: "from client" }]).map((b) => b.text),
    [BASE_SYSTEM, "from client"],
  );
  assert.deepEqual(mergeSystem("   ").map((b) => b.text), [BASE_SYSTEM]);
}

// model routing: which id, and which upstream surface
{
  const r = (m?: string) => {
    const { model, family } = resolveModel(m);
    return `${family}:${model}`;
  };

  // a bare Claude id gets the anthropic. prefix
  assert.equal(r("claude-sonnet-5"), "anthropic:anthropic.claude-sonnet-5");
  assert.equal(r("claude-opus-4-5-20251101"), "anthropic:anthropic.claude-opus-4-5-20251101");
  assert.equal(r("anthropic.claude-opus-5"), "anthropic:anthropic.claude-opus-5");
  assert.equal(r("us.anthropic.claude-opus-5"), "anthropic:us.anthropic.claude-opus-5");

  // any provider.model id passes through untouched
  assert.equal(r("amazon.nova-pro-v1:0"), "openai:amazon.nova-pro-v1:0");
  assert.equal(r("meta.llama3-3-70b-instruct-v1:0"), "openai:meta.llama3-3-70b-instruct-v1:0");
  assert.equal(r("openai.gpt-oss-120b-1:0"), "openai:openai.gpt-oss-120b-1:0");
  assert.equal(r("anthropic.claude-opus-4-6-v1"), "anthropic:anthropic.claude-opus-4-6-v1");

  // foreign ids are not Mantle ids, even the dotted ones → fallback, and the fallback
  // stays inside the family the client asked for
  assert.equal(r("gpt-4o"), "openai:openai.gpt-oss-120b-1:0");
  assert.equal(r("gpt-4.1"), "openai:openai.gpt-oss-120b-1:0");
  assert.equal(r("o3-mini"), "openai:openai.gpt-oss-120b-1:0");
  assert.equal(r("codex-mini-latest"), "openai:openai.gpt-oss-120b-1:0");
  assert.equal(r("gemini-2.5-pro"), "anthropic:anthropic.claude-opus-4-7"); // neither → generic
  assert.equal(r(undefined), "anthropic:anthropic.claude-opus-4-7");
}

// base prompt goes in front of the client's messages on the OpenAI surface too
{
  const msgs = withBaseSystem([
    { role: "system", content: "client rule" },
    { role: "user", content: "hi" },
  ]);
  assert.equal(msgs.length, 3);
  assert.equal(msgs[0]!.role, "system");
  assert.equal(msgs[0]!.content, BASE_SYSTEM);
  assert.equal(msgs[1]!.content, "client rule"); // appended, not replaced
}

// cache_control breakpoints: Anthropic caps them at 4, so ours must yield
{
  assert.equal(countBreakpoints({}), 0);
  assert.equal(countBreakpoints({ system: [{ cache_control: { type: "ephemeral" } }] }), 1);
  assert.equal(
    countBreakpoints({
      system: [{ cache_control: {} }],
      tools: [{ cache_control: {} }, { name: "x" }],
      messages: [{ content: [{ cache_control: {} }, { cache_control: {} }] }],
    }),
    4,
  );

  // under the cap → base block gets cached
  const cached = normalize({ model: "claude-opus-5", messages: [] } as unknown as Params);
  assert.ok((cached.system as Anthropic.TextBlockParam[])[0]!.cache_control);

  // client already used all 4 → ours is dropped rather than 400ing the request
  const full = normalize({
    model: "claude-opus-5",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "a", cache_control: { type: "ephemeral" } },
          { type: "text", text: "b", cache_control: { type: "ephemeral" } },
          { type: "text", text: "c", cache_control: { type: "ephemeral" } },
          { type: "text", text: "d", cache_control: { type: "ephemeral" } },
        ],
      },
    ],
  } as unknown as Params);
  assert.equal((full.system as Anthropic.TextBlockParam[])[0]!.cache_control, undefined);
  assert.equal((full.system as Anthropic.TextBlockParam[])[0]!.text, BASE_SYSTEM); // still sent
}

// model list is a union shape both dialects can read
{
  const list = modelList([{ id: "anthropic.claude-opus-5", created: 1779321600, owned_by: "system" }]);
  assert.equal(list.object, "list"); // OpenAI
  assert.equal(list.has_more, false); // Anthropic
  const m = list.data[0]!;
  assert.equal(m.object, "model");
  assert.equal(m.type, "model");
  assert.equal(m.created, 1779321600);
  assert.equal(m.created_at, "2026-05-21T00:00:00.000Z");
  assert.equal(m.display_name, "anthropic.claude-opus-5");
}

// normalize: merges system, defaults max_tokens, drops sampling only where rejected
{
  // sampling is only stripped for the Claude models that reject it
  const p = normalize({
    model: "claude-opus-5",
    messages: [{ role: "user", content: "hi" }],
    temperature: 0.7,
    top_p: 0.9,
  } as unknown as Params);
  assert.equal(p.model, "anthropic.claude-opus-5");
  assert.equal(p.max_tokens, 16000);
  assert.equal(p.temperature, undefined);
  assert.equal(p.top_p, undefined);

  // gpt-oss accepts sampling, and a gpt-4o request falls back inside its own family
  const oss = normalize({
    model: "gpt-4o",
    messages: [],
    temperature: 0.7,
  } as unknown as Params);
  assert.equal(oss.model, "openai.gpt-oss-120b-1:0");
  assert.equal(oss.temperature, 0.7);
  assert.equal((p.system as Anthropic.TextBlockParam[])[0]!.text, BASE_SYSTEM);

  const old = normalize({
    model: "claude-opus-4-6",
    messages: [],
    temperature: 0.7,
  } as unknown as Params);
  assert.equal(old.temperature, 0.7);
  assert.equal(normalize({ model: "x", messages: [] } as unknown as Params, true).max_tokens, 64000);
}

// OpenAI messages → Anthropic: system hoisted, tool_calls converted,
// consecutive tool results collapsed into one user message
{
  const { system, messages } = splitMessages([
    { role: "system", content: "client rule" },
    { role: "user", content: "weather in paris and london?" },
    {
      role: "assistant",
      content: null,
      tool_calls: [
        { id: "c1", type: "function", function: { name: "w", arguments: '{"city":"paris"}' } },
        { id: "c2", type: "function", function: { name: "w", arguments: "not json" } },
      ],
    },
    { role: "tool", tool_call_id: "c1", content: "18C" },
    { role: "tool", tool_call_id: "c2", content: "12C" },
  ]);

  assert.equal(system, "client rule");
  assert.equal(messages.length, 3);
  const calls = messages[1]!.content as Anthropic.ContentBlockParam[];
  assert.deepEqual(calls.map((b) => b.type), ["tool_use", "tool_use"]);
  assert.deepEqual((calls[0] as Anthropic.ToolUseBlockParam).input, { city: "paris" });
  assert.deepEqual((calls[1] as Anthropic.ToolUseBlockParam).input, {}); // bad JSON degrades, no throw
  const results = messages[2]!.content as Anthropic.ContentBlockParam[];
  assert.equal(messages[2]!.role, "user");
  assert.equal(results.length, 2); // both tool_results in ONE user message
}

// request params
{
  const p = openAIToAnthropic({
    model: "gpt-4o",
    messages: [{ role: "user", content: "hi" }],
    stop: "END",
    reasoning_effort: "minimal",
    tool_choice: "required",
    tools: [{ type: "function", function: { name: "f", parameters: { type: "object" } } }],
    response_format: { type: "json_schema", json_schema: { schema: { type: "object" } } },
  });
  assert.deepEqual(p.stop_sequences, ["END"]);
  assert.deepEqual(p.tool_choice, { type: "any" });
  assert.equal(p.tools?.length, 1);
  assert.equal(p.output_config?.effort, "low");
  assert.equal(p.output_config?.format?.type, "json_schema");
}

// response → OpenAI
{
  const msg = {
    id: "msg_1",
    content: [
      { type: "text", text: "sure" },
      { type: "tool_use", id: "tu_1", name: "f", input: { a: 1 } },
    ],
    stop_reason: "tool_use",
    usage: { input_tokens: 10, output_tokens: 4 },
  } as unknown as Anthropic.Message;

  const out = toChatCompletion(msg, "gpt-4o");
  assert.equal(out.model, "gpt-4o");
  assert.equal(out.choices[0]!.finish_reason, "tool_calls");
  assert.equal(out.choices[0]!.message.content, "sure");
  assert.equal(out.choices[0]!.message.tool_calls?.[0]!.function.arguments, '{"a":1}');
  assert.equal(out.usage.total_tokens, 14);
}

// streaming chunks: tool argument deltas must land on the right tool index
{
  const chunk = makeChunker("gpt-4o");
  const ev = (e: unknown) => chunk(e as Anthropic.MessageStreamEvent);

  const first = ev({
    type: "message_start",
    message: { id: "msg_9", usage: { input_tokens: 7 } },
  })[0] as any;
  assert.equal(first.id, "msg_9");
  assert.equal(first.object, "chat.completion.chunk");
  assert.equal(first.model, "gpt-4o");
  assert.deepEqual(first.choices, [
    { index: 0, delta: { role: "assistant", content: "" }, finish_reason: null },
  ]);
  assert.deepEqual(ev({ type: "ping" }), []);
  assert.deepEqual((ev({ type: "content_block_delta", delta: { type: "text_delta", text: "hi" } })[0] as any).choices[0].delta, { content: "hi" });

  ev({ type: "content_block_start", content_block: { type: "tool_use", id: "tu_1", name: "f" } });
  const second = ev({
    type: "content_block_start",
    content_block: { type: "tool_use", id: "tu_2", name: "g" },
  })[0] as any;
  assert.equal(second.choices[0].delta.tool_calls[0].index, 1);

  const argDelta = ev({
    type: "content_block_delta",
    delta: { type: "input_json_delta", partial_json: '{"x' },
  })[0] as any;
  assert.equal(argDelta.choices[0].delta.tool_calls[0].index, 1);
  assert.equal(argDelta.choices[0].delta.tool_calls[0].function.arguments, '{"x');

  const last = ev({
    type: "message_delta",
    delta: { stop_reason: "tool_use" },
    usage: { output_tokens: 5 },
  })[0] as any;
  assert.equal(last.choices[0].finish_reason, "tool_calls");
  assert.equal(last.usage.total_tokens, 12); // 7 prompt (from message_start) + 5
}

// api key guard
{
  const guard = makeKeyGuard(" sk-a , sk-b ,, ");
  assert.equal(guard.required, true);
  assert.equal(guard.accepts("sk-a"), true);
  assert.equal(guard.accepts("sk-b"), true);
  assert.equal(guard.accepts("sk-c"), false);
  assert.equal(guard.accepts("sk-a "), false); // no sloppy trailing-space match
  assert.equal(guard.accepts("sk-"), false); // no prefix match
  assert.equal(guard.accepts(""), false);
  assert.equal(guard.accepts("sk-a-longer-than-any-key"), false); // length mismatch must not throw

  const open = makeKeyGuard(undefined);
  assert.equal(open.required, false);
  assert.equal(open.accepts(""), true); // unset = auth skipped
  assert.equal(makeKeyGuard("   ").required, false);

  assert.equal(presentedKey("Bearer sk-a"), "sk-a");
  assert.equal(presentedKey("bearer  sk-a "), "sk-a");
  assert.equal(presentedKey(undefined, " sk-b "), "sk-b");
  assert.equal(presentedKey("", "sk-b"), "sk-b");
  assert.equal(presentedKey("Bearer sk-a", "sk-b"), "sk-a"); // bearer wins
  assert.equal(presentedKey(), "");
}

// rate limiter
{
  const exceeded = makeRateLimiter(3);
  const t = 1_000_000;
  assert.equal(exceeded("k", t), false);
  assert.equal(exceeded("k", t), false);
  assert.equal(exceeded("k", t), false);
  assert.equal(exceeded("k", t), true); // 4th in the window
  assert.equal(exceeded("other", t), false); // buckets are per key
  assert.equal(exceeded("k", t + 60_001), false); // window rolled over
  assert.equal(makeRateLimiter(0)("k", t), false); // 0 disables
}

// error redaction: account id and ARNs must not reach the client
{
  const raw =
    'User: arn:aws:iam::029046030337:user/BedrockAPIKey-ucpa is not authorized on arn:aws:bedrock-mantle:us-east-1:029046030337:project/default';
  const safe = redact(raw);
  assert.ok(!safe.includes("029046030337"), "account id leaked");
  assert.ok(!safe.includes("BedrockAPIKey-ucpa"), "IAM user leaked");
  assert.ok(safe.includes("arn:aws:***"));
  assert.equal(redact("plain message"), "plain message");
}


// ─── Converse translation ──────────────────────────────────────────────────────

// request: system blocks, tool_result split, cachePoint gating, thinking passthrough
{
  const p = {
    model: "anthropic.claude-opus-4-7",
    max_tokens: 512,
    system: [
      { type: "text", text: "BASE", cache_control: { type: "ephemeral" } },
      { type: "text", text: "CLIENT" },
    ],
    thinking: { type: "adaptive" },
    stop_sequences: ["END"],
    tool_choice: { type: "any" },
    tools: [{ name: "w", description: "weather", input_schema: { type: "object" } }],
    messages: [
      { role: "user", content: "weather?" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "hmm", signature: "sig" },
          { type: "text", text: "checking" },
          { type: "tool_use", id: "tu_1", name: "w", input: { city: "paris" } },
        ],
      },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "tu_1", content: "18C" }] },
      { role: "system", content: "operator rule" },
    ],
  } as never;

  const withCaching = messagesToConverse(p, true);
  assert.deepEqual(withCaching.system, [
    { text: "BASE" },
    { cachePoint: { type: "default" } }, // cache_control became a cachePoint
    { text: "CLIENT" },
    { text: "operator rule" }, // a mid-conversation system message joins system, not user
  ]);
  assert.deepEqual(withCaching.additionalModelRequestFields, { thinking: { type: "adaptive" } });
  assert.deepEqual(withCaching.inferenceConfig, { maxTokens: 512, stopSequences: ["END"] });
  assert.deepEqual(withCaching.toolConfig?.toolChoice, { any: {} });
  assert.equal(withCaching.toolConfig?.tools[0]!.toolSpec.name, "w");

  // Llama and gpt-oss reject cachePoint outright, so it must vanish when unsupported
  const noCaching = messagesToConverse(p, false);
  assert.deepEqual(noCaching.system, [
    { text: "BASE" },
    { text: "CLIENT" },
    { text: "operator rule" },
  ]);

  // roles: system messages are hoisted out, so only user/assistant remain
  assert.deepEqual(withCaching.messages.map((m) => m.role), ["user", "assistant", "user"]);
  const assistant = withCaching.messages[1]!.content;
  assert.deepEqual(assistant.map((b) => Object.keys(b)[0]), ["text", "toolUse"]); // thinking dropped
  assert.deepEqual(withCaching.messages[2]!.content, [
    { toolResult: { toolUseId: "tu_1", content: [{ text: "18C" }] } },
  ]);
}

// response
{
  const msg = converseToMessage(
    {
      output: {
        message: {
          content: [{ text: "here" }, { toolUse: { toolUseId: "t1", name: "w", input: { c: 1 } } }],
        },
      },
      stopReason: "tool_use",
      usage: { inputTokens: 11, outputTokens: 3 },
    },
    "anthropic.claude-opus-4-7",
  );
  assert.equal(msg.type, "message");
  assert.equal(msg.stop_reason, "tool_use");
  assert.deepEqual(msg.content.map((b) => b.type), ["text", "tool_use"]);
  assert.deepEqual((msg.content[1] as Anthropic.ToolUseBlock).input, { c: 1 });
  assert.equal(msg.usage.input_tokens, 11);

  // an empty answer still has to produce one block
  assert.equal(converseToMessage({ output: { message: { content: [] } } }, "m").content.length, 1);
}

// stream: Converse omits contentBlockStart for text, Anthropic clients require it
{
  const e = makeConverseEmitter("anthropic.claude-opus-4-7");
  const events = [
    ...e.push("messageStart", { role: "assistant" }),
    ...e.push("contentBlockDelta", { contentBlockIndex: 0, delta: { text: "Hi" } }),
    ...e.push("contentBlockDelta", { contentBlockIndex: 0, delta: { text: "!" } }),
    ...e.push("contentBlockStop", { contentBlockIndex: 0 }),
    ...e.push("contentBlockStart", {
      contentBlockIndex: 1,
      start: { toolUse: { toolUseId: "t1", name: "w" } },
    }),
    ...e.push("contentBlockDelta", { contentBlockIndex: 1, delta: { toolUse: { input: '{"c":1}' } } }),
    ...e.push("contentBlockStop", { contentBlockIndex: 1 }),
    ...e.push("messageStop", { stopReason: "tool_use" }),
    ...e.push("metadata", { usage: { inputTokens: 9, outputTokens: 4 } }),
    ...e.finish(),
  ];

  assert.deepEqual(events.map((x) => x.type), [
    "message_start",
    "content_block_start", // synthesised on the first text delta
    "content_block_delta",
    "content_block_delta",
    "content_block_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
  ]);

  // one block open at a time, every stop matching its start
  let open: number | null = null;
  for (const x of events) {
    if (x.type === "content_block_start") {
      assert.equal(open, null, "a block opened while another was open");
      open = x.index as number;
    }
    if (x.type === "content_block_stop") {
      assert.equal(open, x.index);
      open = null;
    }
  }
  assert.equal(open, null);

  const delta = events.at(-2) as unknown as {
    delta: { stop_reason: string };
    usage: { input_tokens: number; output_tokens: number };
  };
  assert.equal(delta.delta.stop_reason, "tool_use");
  assert.deepEqual(delta.usage, { input_tokens: 9, output_tokens: 4 });

  // an unterminated stream still gets closed properly
  const e2 = makeConverseEmitter("m");
  e2.push("messageStart", {});
  e2.push("contentBlockDelta", { contentBlockIndex: 0, delta: { text: "partial" } });
  assert.deepEqual(e2.finish().map((x) => x.type), [
    "content_block_stop",
    "message_delta",
    "message_stop",
  ]);
}

// catalogue: profile ARNs carry no slash before "foundation-model", which silently
// dropped every INFERENCE_PROFILE model until it was fixed
{
  const cat = buildCatalogue(
    {
      modelSummaries: [
        { modelId: "amazon.nova-lite-v1:0", inferenceTypesSupported: ["ON_DEMAND"], outputModalities: ["TEXT"] },
        { modelId: "anthropic.claude-opus-4-6-v1", inferenceTypesSupported: ["INFERENCE_PROFILE"], outputModalities: ["TEXT"] },
        { modelId: "meta.llama-x", inferenceTypesSupported: ["INFERENCE_PROFILE"], outputModalities: ["TEXT"] },
        { modelId: "amazon.titan-embed", inferenceTypesSupported: ["ON_DEMAND"], outputModalities: ["EMBEDDING"] },
        { modelId: "some.provisioned-only", inferenceTypesSupported: ["PROVISIONED"], outputModalities: ["TEXT"] },
      ],
    },
    {
      inferenceProfileSummaries: [
        {
          inferenceProfileId: "us.anthropic.claude-opus-4-6-v1",
          status: "ACTIVE",
          models: [{ modelArn: "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-4-6-v1" }],
        },
        {
          inferenceProfileId: "eu.meta.llama-x",
          status: "INACTIVE", // ignored, so meta.llama-x is not callable
          models: [{ modelArn: "arn:aws:bedrock:eu-west-1::foundation-model/meta.llama-x" }],
        },
      ],
    },
  );

  assert.equal(cat.get("amazon.nova-lite-v1:0")?.upstreamId, "amazon.nova-lite-v1:0");
  assert.equal(
    cat.get("anthropic.claude-opus-4-6-v1")?.upstreamId,
    "us.anthropic.claude-opus-4-6-v1",
  );
  assert.equal(cat.has("meta.llama-x"), false); // no ACTIVE profile
  assert.equal(cat.has("amazon.titan-embed"), false); // not a text model
  assert.equal(cat.has("some.provisioned-only"), false); // not callable on demand

  // caching support is measured per model, not assumed
  assert.equal(cat.get("anthropic.claude-opus-4-6-v1")?.caching, true);
  assert.equal(cat.get("amazon.nova-lite-v1:0")?.caching, true);
}

// event-stream frame parser
{
  const frame = (event: string, payload: unknown) => {
    const body = Buffer.from(JSON.stringify(payload));
    const name = Buffer.from(":event-type");
    // [1B name len][name][1B type=7][2B value len][value]
    const header = Buffer.concat([
      Buffer.from([name.length]),
      name,
      Buffer.from([7]),
      (() => { const b = Buffer.alloc(2); b.writeUInt16BE(event.length); return b; })(),
      Buffer.from(event),
    ]);
    const total = 12 + header.length + body.length + 4;
    const out = Buffer.alloc(total);
    out.writeUInt32BE(total, 0);
    out.writeUInt32BE(header.length, 4);
    header.copy(out, 12);
    body.copy(out, 12 + header.length);
    return out;
  };

  const wire = Buffer.concat([
    frame("messageStart", { role: "assistant" }),
    frame("contentBlockDelta", { contentBlockIndex: 0, delta: { text: "hi" } }),
    frame("messageStop", { stopReason: "end_turn" }),
  ]);

  // split mid-frame to prove the reader buffers across chunks
  const cuts = [7, 40, wire.length];
  let from = 0;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const to of cuts) {
        controller.enqueue(new Uint8Array(wire.subarray(from, to)));
        from = to;
      }
      controller.close();
    },
  });

  const seen: string[] = [];
  for await (const f of eventStreamFrames(body)) seen.push(f.event);
  assert.deepEqual(seen, ["messageStart", "contentBlockDelta", "messageStop"]);
}

console.log("ok");

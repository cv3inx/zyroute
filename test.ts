import assert from "node:assert/strict";
import type Anthropic from "@anthropic-ai/sdk";
import { makeKeyGuard, makeRateLimiter, presentedKey } from "./src/auth.js";
import { redact } from "./src/log.js";
import {
  chatToMessage,
  makeMessageEmitter,
  messagesToChat,
  sseJSON,
} from "./src/anthropic-to-openai.js";
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
} from "./src/translate.js";

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

  // anthropic family → Messages API
  assert.equal(r("claude-sonnet-5"), "anthropic:anthropic.claude-sonnet-5");
  assert.equal(r("claude-opus-4-5-20251101"), "anthropic:anthropic.claude-opus-4-5-20251101");
  assert.equal(r("anthropic.claude-opus-5"), "anthropic:anthropic.claude-opus-5");
  assert.equal(r("us.anthropic.claude-opus-5"), "anthropic:us.anthropic.claude-opus-5");

  // every other Mantle model → OpenAI chat completions, id untouched
  assert.equal(r("openai.gpt-oss-120b"), "openai:openai.gpt-oss-120b");
  assert.equal(r("openai.gpt-5.5"), "openai:openai.gpt-5.5");
  assert.equal(r("qwen.qwen3-235b"), "openai:qwen.qwen3-235b");
  assert.equal(r("google.gemma-4-31b"), "openai:google.gemma-4-31b");
  assert.equal(r("moonshotai.kimi-k2.5"), "openai:moonshotai.kimi-k2.5");

  // foreign ids are not Mantle ids, even the dotted ones → fallback, and the fallback
  // stays inside the family the client asked for
  assert.equal(r("gpt-4o"), "openai:openai.gpt-oss-120b");
  assert.equal(r("gpt-4.1"), "openai:openai.gpt-oss-120b");
  assert.equal(r("o3-mini"), "openai:openai.gpt-oss-120b");
  assert.equal(r("codex-mini-latest"), "openai:openai.gpt-oss-120b");
  assert.equal(r("gemini-2.5-pro"), "anthropic:anthropic.claude-opus-5"); // neither → generic
  assert.equal(r(undefined), "anthropic:anthropic.claude-opus-5");
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
  assert.equal(oss.model, "openai.gpt-oss-120b");
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

// ─── bridge: Anthropic dialect → OpenAI upstream ──────────────────────────────

// request translation, including the tool_result → role:tool split
{
  const chat = messagesToChat(
    {
      model: "openai.gpt-oss-120b",
      max_tokens: 512,
      system: [
        { type: "text", text: "BASE" },
        { type: "text", text: "CLIENT" },
      ],
      stop_sequences: ["END"],
      output_config: { effort: "xhigh" },
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
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "tu_1", content: "18C" },
            { type: "text", text: "and london?" },
          ],
        },
      ],
    } as never,
    "openai.gpt-oss-120b",
  );

  assert.equal(chat.model, "openai.gpt-oss-120b");
  assert.equal(chat.max_tokens, 512);
  assert.deepEqual(chat.stop, ["END"]);
  assert.equal(chat.reasoning_effort, "high"); // xhigh has no OpenAI equivalent
  assert.equal(chat.tool_choice, "required");
  assert.deepEqual(chat.tools?.[0]!.function.parameters, { type: "object" });

  assert.deepEqual(
    chat.messages.map((m) => m.role),
    ["system", "user", "assistant", "tool", "user"],
  );
  assert.equal(chat.messages[0]!.content, "BASE\n\nCLIENT"); // base first, client appended
  const assistant = chat.messages[2]!;
  assert.equal(assistant.content, "checking"); // thinking block dropped
  assert.equal(assistant.tool_calls?.[0]!.function.arguments, '{"city":"paris"}');
  assert.equal(chat.messages[3]!.tool_call_id, "tu_1");

  // streaming is an explicit argument — inferring it from p.stream silently sent a
  // non-streaming request upstream while the client waited on SSE
  assert.equal(chat.stream, undefined);
  const streamed = messagesToChat({ model: "m", max_tokens: 1, messages: [] } as never, "m", true);
  assert.equal(streamed.stream, true);
  assert.deepEqual(streamed.stream_options, { include_usage: true });
}

// response translation
{
  const msg = chatToMessage(
    {
      id: "cc_1",
      choices: [
        {
          message: {
            content: "here",
            tool_calls: [{ id: "call_1", function: { name: "w", arguments: '{"city":"paris"}' } }],
          },
          finish_reason: "tool_calls",
        },
      ],
      usage: { prompt_tokens: 9, completion_tokens: 3 },
    },
    "openai.gpt-oss-120b",
  );

  assert.equal(msg.id, "msg_cc_1");
  assert.equal(msg.type, "message");
  assert.equal(msg.stop_reason, "tool_use");
  assert.deepEqual(msg.content.map((b) => b.type), ["text", "tool_use"]);
  assert.deepEqual((msg.content[1] as Anthropic.ToolUseBlock).input, { city: "paris" });
  assert.equal(msg.usage.input_tokens, 9);

  // an empty completion still has to produce one block
  const empty = chatToMessage({ choices: [{ message: { content: "" } }] }, "m");
  assert.equal(empty.content.length, 1);
  assert.equal(empty.stop_reason, "end_turn");
}

// SSE: OpenAI chunks → a well-formed Anthropic event sequence
{
  const emitter = makeMessageEmitter("openai.gpt-oss-120b");
  const events = [
    ...emitter.push({ id: "cc_9", usage: { prompt_tokens: 11 }, choices: [{ delta: { content: "He" } }] }),
    ...emitter.push({ choices: [{ delta: { content: "llo" } }] }),
    ...emitter.push({
      choices: [{ delta: { tool_calls: [{ index: 0, id: "call_1", function: { name: "w", arguments: "" } }] } }],
    }),
    ...emitter.push({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{"c":1}' } }] } }] }),
    ...emitter.push({ choices: [{ delta: {}, finish_reason: "tool_calls" }], usage: { completion_tokens: 7 } }),
    ...emitter.finish(),
  ];

  assert.deepEqual(
    events.map((e) => e.type),
    [
      "message_start",
      "content_block_start",
      "content_block_delta",
      "content_block_delta",
      "content_block_stop", // text closes before the tool block opens
      "content_block_start",
      "content_block_delta",
      "content_block_stop",
      "message_delta",
      "message_stop",
    ],
  );

  // exactly one block open at a time, and indices are sequential from 0
  let open: number | null = null;
  const opened: number[] = [];
  for (const e of events) {
    if (e.type === "content_block_start") {
      assert.equal(open, null, "a block opened while another was still open");
      open = e.index as number;
      opened.push(open);
    }
    if (e.type === "content_block_stop") {
      assert.equal(open, e.index, "content_block_stop index does not match the open block");
      open = null;
    }
  }
  assert.equal(open, null, "stream ended with a block still open");
  assert.deepEqual(opened, [0, 1]);

  const start = events[0] as unknown as { message: { id: string; usage: { input_tokens: number } } };
  assert.equal(start.message.id, "msg_cc_9");
  assert.equal(start.message.usage.input_tokens, 11);

  const delta = events.at(-2) as unknown as { delta: { stop_reason: string }; usage: { output_tokens: number } };
  assert.equal(delta.delta.stop_reason, "tool_use");
  assert.equal(delta.usage.output_tokens, 7);

  const toolStart = events[5] as unknown as { content_block: { id: string; name: string } };
  assert.equal(toolStart.content_block.id, "call_1");
  assert.equal(toolStart.content_block.name, "w");
}

// SSE parser: partial lines across reads, [DONE], malformed chunks
{
  const wire = [
    'data: {"id":"a"}\n\n',
    'data: {"id":"b"',
    '}\n\ndata: not-json\n\n',
    'data: {"id":"c"}\n\ndata: [DONE]\n\ndata: {"id":"never"}\n\n',
  ];
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const piece of wire) controller.enqueue(new TextEncoder().encode(piece));
      controller.close();
    },
  });

  const seen: string[] = [];
  for await (const chunk of sseJSON(body)) seen.push((chunk as { id: string }).id);
  assert.deepEqual(seen, ["a", "b", "c"]); // split chunk rejoined, junk skipped, DONE stops it
}

console.log("ok");

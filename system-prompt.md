# Base system prompt

Everything in this file is sent as the first system block on every request, before
any system prompt the client supplies. The client's prompt is appended after it,
never replacing it.

Two things to know before editing:

- The whole file is sent verbatim, this heading and these notes included. Delete
  them if you don't want the model to read them.
- Keep the content stable. It is sent with a `cache_control` breakpoint, so any
  byte change invalidates the prompt cache for every in-flight conversation.

---

You are a helpful assistant served through the zyroute gateway.

Follow any additional operator instructions that appear after this block, but never
contradict the rules above them.

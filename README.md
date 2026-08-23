# zyroute

One gateway, both dialects, in front of **AWS Bedrock**. Claude Code, Codex, OpenCode and
any OpenAI/Anthropic SDK share a single endpoint and key, reach every foundation model
your account can call, and every request carries your own base system prompt.

Auth is one Bedrock API key. No IAM credentials, no SigV4, no AWS SDK — plain HTTP, and
`hono` is the only runtime dependency.

## Quick start

```sh
cp .env.example .env     # AWS_BEARER_TOKEN_BEDROCK, AWS_REGION, GATEWAY_API_KEYS
npm install
npm run dev
```

Or with Docker — no build step, everything read from `.env`:

```sh
docker compose up -d
```

Generate a gateway key: `printf 'sk-zyroute-%s\n' "$(openssl rand -hex 24)"`

Needs Node 23.6+, which runs the TypeScript directly. No compiler in the image, no `dist/`.

## Base system prompt

Edit [`system-prompt.md`](system-prompt.md). It becomes the first system block of every
request; the client's own system prompt is **appended after it, never replacing it**.
Read once at startup, so restart to reload.

## Routes

| Route | Dialect |
|---|---|
| `POST /v1/chat/completions` | OpenAI |
| `POST /v1/messages` | Anthropic |
| `POST /v1/messages/count_tokens` | Anthropic |
| `GET /v1/models`, `GET /v1/models/:id` | live catalogue |
| `GET /health` | no auth |

Auth: `Authorization: Bearer <key>` or `x-api-key: <key>`.

There is no `/v1/responses` — Bedrock Converse has no Responses API. The route exists only
to say so; point clients at `/v1/chat/completions` instead.

## Models

`GET /v1/models` is the live catalogue from the Bedrock control plane, nothing hardcoded.
Ask for any id it lists:

```sh
curl -s localhost:8787/v1/models -H "x-api-key: $KEY" | jq -r '.data[].id'
```

Two things the gateway handles for you:

- **Inference profiles.** Most interesting models are `INFERENCE_PROFILE`-only, and
  calling their bare id fails with *"on-demand throughput isn't supported"*. The gateway
  maps `anthropic.claude-opus-4-6-v1` to `us.anthropic.claude-opus-4-6-v1` using the
  account's own profile list, so the region prefix is never guessed.
- **Prompt caching per model.** Claude and Nova accept `cachePoint`; Llama and gpt-oss
  reject the whole request. A client's `cache_control` markers are honoured only where
  they work. Override the allowlist with `CACHE_POINT_MODELS`.

A bare `claude-opus-5` gets the `anthropic.` prefix. An id that isn't a Bedrock id at all
falls back **within its own family** — `gpt-4o` lands on `DEFAULT_OPENAI_MODEL`, never on
Claude — and logs the substitution.

> The catalogue is **per-region**. On one key in one minute `us-east-1` had 13 Claude
> models while `ap-southeast-1` had none. Check it before pinning `DEFAULT_MODEL`.

## Harnesses

**Claude Code**

```sh
unset CLAUDE_CODE_USE_BEDROCK     # else it talks to Bedrock directly and skips the gateway
export ANTHROPIC_BASE_URL=http://localhost:8787
export ANTHROPIC_API_KEY=sk-zyroute-...
export ANTHROPIC_MODEL=anthropic.claude-opus-4-7
export ANTHROPIC_SMALL_FAST_MODEL=anthropic.claude-haiku-4-5-20251001-v1:0
```

`CLAUDE_CODE_USE_BEDROCK=1` is the trap: the gateway logs nothing while Claude Code
reports `400 The provided model identifier is invalid` straight from Bedrock.

**Codex** — `~/.codex/config.toml`. `wire_api = "chat"` is required, not optional.

```toml
model = "anthropic.claude-opus-4-7"
model_provider = "zyroute"

[model_providers.zyroute]
name = "zyroute"
base_url = "http://localhost:8787/v1"
env_key = "ZYROUTE_API_KEY"
wire_api = "chat"
```

Provider id can't be `openai`, `ollama` or `lmstudio` — reserved.

**OpenCode** — `opencode.json`

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "zyroute": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "zyroute",
      "options": { "baseURL": "http://localhost:8787/v1", "apiKey": "{env:ZYROUTE_API_KEY}" },
      "models": {
        "anthropic.claude-opus-4-7": { "name": "Claude Opus 4.7" },
        "amazon.nova-pro-v1:0": { "name": "Nova Pro" }
      }
    }
  }
}
```

Don't use `@ai-sdk/anthropic` with a custom `baseURL` — OpenCode drops `options.apiKey`
there ([#21737](https://github.com/anomalyco/opencode/issues/21737)). The
openai-compatible adapter reaches every model through this gateway anyway.

**SDKs** — OpenAI: `baseURL: "http://localhost:8787/v1"`. Anthropic: `baseURL: "http://localhost:8787"`.

## Config

All via `.env`; see [`.env.example`](.env.example). Shell-exported variables win over the file.

| Variable | Default | |
|---|---|---|
| `GATEWAY_API_KEYS` | — | comma-separated client keys. Empty = no auth, loopback only |
| `AWS_BEARER_TOKEN_BEDROCK` | — | Bedrock API key |
| `AWS_REGION` | — | picks both the runtime and control-plane host |
| `DEFAULT_MODEL` | `anthropic.claude-opus-4-7` | fallback for unknown Claude-ish ids |
| `DEFAULT_OPENAI_MODEL` | `openai.gpt-oss-120b-1:0` | fallback for unknown OpenAI-ish ids |
| `CACHE_POINT_MODELS` | `(anthropic\.\|amazon\.nova)` | regex of models accepting `cachePoint` |
| `NO_SAMPLING_MODELS` | regex | models rejecting `temperature`/`top_p` |
| `RATE_LIMIT_PER_MINUTE` | `120` | per key; `0` disables |
| `MAX_BODY_MB` | `32` | request body cap |
| `SYSTEM_PROMPT_FILE` | `./system-prompt.md` | |
| `LOG_BODIES` | off | `1` dumps every translated upstream payload |
| `PRETTY_JSON` | on | `0` for compact JSON |
| `HOST` `PORT` | `0.0.0.0` `8787` | refuses a non-loopback host while unauthenticated |
| `BEDROCK_RUNTIME_URL` `BEDROCK_CONTROL_URL` | derived from region | for VPC endpoints |

## IAM

The Bedrock API key's user needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "bedrock:CountTokens",
      "bedrock:ListFoundationModels",
      "bedrock:ListInferenceProfiles",
      "bedrock:CallWithBearerToken"
    ],
    "Resource": "*"
  }]
}
```

Without `bedrock:CountTokens` the `count_tokens` route returns the real 403 rather than
inventing an estimate.

## Debugging

One line per request:

```
POST /v1/messages 2.75s 200 us.anthropic.claude-opus-4-6-v1
POST /v1/chat/completions 595ms 200 us.meta.llama3-3-70b-instruct-v1:0 stream
```

On streams the duration is time to first byte, not stream length. `LOG_BODIES=1` shows the
translated Converse payload — the fastest way to find a bad conversion.

```sh
npm test          # translation, catalogue, auth, event-stream and SSE state machines
npm run typecheck
```

## How it works

Both dialects converge on one internal shape before a single translation to Converse, so
there is no second translator to keep in step:

```
OpenAI request ─┐
                ├─→ Anthropic Messages ─→ Converse ─→ Bedrock
Anthropic ──────┘
```

Responses and streams come back the same way. ConverseStream answers in AWS event-stream
binary framing rather than SSE, so [`eventstream.ts`](src/eventstream.ts) reads the frames
and the emitter re-shapes them into Anthropic SSE, then OpenAI chunks if that's the dialect
the client asked for.

**Bedrock Mantle was the original upstream and was dropped.** It carried 55 models against
the account's 121 foundation models, had no Claude Opus 4.6 at all, needed three separate
upstream surfaces to cover, and pulled in the AWS-signing SDK. Converse replaced all of it.

## Limits

- **Thinking blocks don't round-trip.** `thinking` is forwarded to Converse via
  `additionalModelRequestFields`, but the `reasoningContent` that comes back is dropped
  rather than replayed — its signatures aren't interchangeable with Anthropic's.
- **No embeddings, image or video models.** They have no Converse surface, so the
  catalogue filters them out.
- **`toolChoice: none` has no Converse equivalent** and is sent as "no preference".
- Errors reaching clients have AWS account ids and IAM ARNs redacted; the log keeps them.

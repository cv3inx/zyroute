# zyroute

One gateway, both dialects, in front of **AWS Bedrock Mantle**. Claude Code, Codex,
OpenCode and any OpenAI/Anthropic SDK share a single endpoint and key — and every
request carries your own base system prompt.

## Quick start

```sh
cp .env.example .env     # fill in AWS_BEARER_TOKEN_BEDROCK, AWS_REGION, GATEWAY_API_KEYS
npm install
npm run dev
```

Or with Docker — no build step, everything read from `.env`:

```sh
docker compose up -d
```

Generate a gateway key: `printf 'sk-zyroute-%s\n' "$(openssl rand -hex 24)"`

Needs Node 23.6+, which runs the TypeScript directly. There is no compiler in the
image and no `dist/`.

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
| `POST /v1/responses` | OpenAI Responses |
| `GET /v1/models`, `GET /v1/models/:id` | live Bedrock catalogue |
| `GET /health` | no auth |

Auth: `Authorization: Bearer <key>` or `x-api-key: <key>`.

## Models

Mantle namespaces models `provider.model` and serves them on two upstream surfaces.
The gateway picks the surface from the id and translates when the client's dialect
doesn't match:

| Model | Upstream | from OpenAI client | from Anthropic client |
|---|---|---|---|
| `anthropic.claude-*` | `/anthropic/v1/messages` | translated | passthrough |
| everything else | `/v1/chat/completions` | passthrough | translated |

`claude-opus-5` gets the `anthropic.` prefix added. An id that isn't a Mantle id falls
back **within its own family** — `gpt-4o` lands on `DEFAULT_OPENAI_MODEL`, never on Claude.

> The catalogue is **per-region and changes over time**. Measured on one key in one
> minute: `us-east-1` had 55 models incl. 6 Claude; `us-west-2` 48 with only haiku;
> `ap-southeast-1` and `eu-central-1` had no Claude at all. Check `GET /v1/models`
> before pinning `DEFAULT_MODEL`, and again if you hit `not_found_error`.

## Harnesses

**Claude Code**

```sh
unset CLAUDE_CODE_USE_BEDROCK     # else it talks to Bedrock directly and skips the gateway
export ANTHROPIC_BASE_URL=http://localhost:8787
export ANTHROPIC_API_KEY=sk-zyroute-...
export ANTHROPIC_MODEL=claude-opus-5
export ANTHROPIC_SMALL_FAST_MODEL=claude-haiku-4-5
```

`CLAUDE_CODE_USE_BEDROCK=1` is the trap: the gateway logs nothing while Claude Code
reports `400 The provided model identifier is invalid` from Bedrock. Also pick a model
`GET /v1/models` actually lists — dated ids like `claude-sonnet-4-5-20250929` are not
on Mantle, and the gateway returns a clean 404 rather than guessing a substitute.

**Codex** — `~/.codex/config.toml`

```toml
model = "openai.gpt-oss-120b"
model_provider = "zyroute"

[model_providers.zyroute]
name = "zyroute"
base_url = "http://localhost:8787/v1"
env_key = "ZYROUTE_API_KEY"
wire_api = "chat"      # "responses" also works, but reaches fewer models
```

Provider id can't be `openai`, `ollama` or `lmstudio` — those are reserved.

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
        "openai.gpt-oss-120b": { "name": "GPT-OSS 120B" },
        "anthropic.claude-opus-5": { "name": "Claude Opus 5" }
      }
    }
  }
}
```

`@ai-sdk/openai-compatible` reaches **both** families here, Claude included. Don't use
`@ai-sdk/anthropic` with a custom `baseURL` — OpenCode drops `options.apiKey` there
([#21737](https://github.com/anomalyco/opencode/issues/21737)).

**SDKs** — OpenAI: `baseURL: "http://localhost:8787/v1"`. Anthropic: `baseURL: "http://localhost:8787"`.

## Config

All of it via `.env`; see [`.env.example`](.env.example). Shell-exported variables win
over the file.

| Variable | Default | |
|---|---|---|
| `GATEWAY_API_KEYS` | — | comma-separated client keys. Empty = no auth, loopback only |
| `AWS_BEARER_TOKEN_BEDROCK` | — | Bedrock API key. Required for non-Claude models |
| `AWS_REGION` | — | or use SigV4 via the standard AWS variables |
| `DEFAULT_MODEL` | `anthropic.claude-opus-5` | fallback for unknown Claude-ish ids |
| `DEFAULT_OPENAI_MODEL` | `openai.gpt-oss-120b` | fallback for unknown OpenAI-ish ids |
| `RATE_LIMIT_PER_MINUTE` | `120` | per key; `0` disables |
| `MAX_BODY_MB` | `32` | request body cap |
| `NO_SAMPLING_MODELS` | regex | models that reject `temperature`/`top_p` |
| `SYSTEM_PROMPT_FILE` | `./system-prompt.md` | |
| `LOG_BODIES` | off | `1` dumps every translated upstream payload |
| `PRETTY_JSON` | on | `0` for compact JSON |
| `HOST` `PORT` | `0.0.0.0` `8787` | refuses a non-loopback host while unauthenticated |

## Debugging

One line per request:

```
POST /v1/messages 955ms 200 anthropic.claude-opus-5 → /anthropic/v1/messages
POST /v1/messages 503ms 200 openai.gpt-oss-20b → /v1/chat/completions stream
```

On streams the duration is time to first byte, not stream length. `LOG_BODIES=1` shows
the translated upstream payload, which is the fastest way to find a bad conversion.

```sh
npm test          # translation, auth, rate limit, SSE state machine
npm run typecheck
```

## Limits

- **`count_tokens` needs IAM.** A key without `bedrock-mantle:CountTokens` gets 403;
  the gateway forwards the real error instead of inventing an estimate. Grant:

  ```json
  { "Effect": "Allow",
    "Action": ["bedrock-mantle:CreateInference", "bedrock-mantle:CountTokens",
               "bedrock-mantle:ListModels", "bedrock-mantle:CallWithBearerToken"],
    "Resource": "*" }
  ```

- **Non-Claude models need the bearer token.** SigV4 is only implemented on the
  Anthropic surface, by the Mantle SDK.
- **Thinking is dropped when bridging.** An Anthropic client on a non-Anthropic model
  loses `thinking` blocks — that reasoning isn't interchangeable and carries no
  signature to replay.
- Some models need a Marketplace subscription (`openai.gpt-5.5`), and not every listed
  model is served on every route. Upstream says so plainly.
- Mantle has no embeddings, completions, batch or files endpoints.
- Errors reaching clients have AWS account ids and IAM ARNs redacted; the log keeps them.

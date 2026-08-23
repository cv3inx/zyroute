# zyroute

One gateway that speaks both the OpenAI and Anthropic dialects in front of **AWS Bedrock
Mantle**, so Claude Code, Codex, and any OpenAI/Anthropic SDK client can share a single
endpoint and a single key — and every request carries your own base system prompt.

## Setup

```sh
npm install
cp .env.example .env      # fill in AWS_BEARER_TOKEN_BEDROCK, AWS_REGION, GATEWAY_API_KEYS
npm run dev
```

Generate a gateway key:

```sh
printf 'sk-zyroute-%s\n' "$(openssl rand -hex 24)"
```

`.env` is read with Node's `--env-file-if-exists`, which **does not overwrite variables
already exported in your shell**. If a value looks ignored, check `env | grep AWS` first.

## The base system prompt

Edit [`system-prompt.md`](system-prompt.md). It is sent as the first system block on
every request and the client's own system prompt is **appended after it, never
replacing it**. The file is read once at startup, so restart after editing.

The whole file is sent verbatim, including any headings you leave in it.

## Routes

| Route | Dialect |
|---|---|
| `POST /v1/chat/completions` | OpenAI |
| `POST /v1/messages` | Anthropic |
| `POST /v1/messages/count_tokens` | Anthropic |
| `POST /v1/responses` | OpenAI Responses |
| `GET /v1/models`, `GET /v1/models/:id` | live Bedrock catalogue, both shapes |
| `GET /health` | unauthenticated |

Auth: `Authorization: Bearer <key>` or `x-api-key: <key>`, matched in constant time
against `GATEWAY_API_KEYS`. With no keys configured the gateway skips auth and binds to
`127.0.0.1` only — and it **refuses to start** on a non-loopback `HOST` while unauthenticated.

Per-key limits, because anyone holding a gateway key spends from your AWS account:
`RATE_LIMIT_PER_MINUTE` (default 120, `0` disables it) and `MAX_BODY_MB` (default 32).
Upstream error text is forwarded to clients with AWS account ids and IAM ARNs redacted;
the full text stays in the server log.

## Model routing

Mantle namespaces models as `provider.model` and serves them on two different upstream
surfaces. The gateway picks the surface from the id, and translates when the client's
dialect doesn't match it:

| Model | Upstream | OpenAI client | Anthropic client |
|---|---|---|---|
| `anthropic.claude-*` | `/anthropic/v1/messages` | translated | passthrough |
| everything else | `/v1/chat/completions` | passthrough | translated |

A bare `claude-opus-5` gets the `anthropic.` prefix. An id that isn't a Mantle id at all
falls back and logs a warning — and the **fallback stays inside the family the client
asked for**: `gpt-4o` lands on `DEFAULT_OPENAI_MODEL`, never on Claude. Swapping families
would change tokenizer, tool semantics and price at once.

`GET /v1/models` returns the live catalogue — nothing is hardcoded. Not every listed
model is served on every route; upstream says so plainly when you pick one that isn't.

### The catalogue depends on your region

Measured on one account, same key, same minute:

| Region | Models | Anthropic models |
|---|---|---|
| `us-east-1` | 55 | opus-5, opus-4-8, opus-4-7, sonnet-5, fable-5, haiku-4-5 |
| `us-west-2` | 48 | haiku-4-5 only |
| `ap-southeast-1` | 38 | none |
| `eu-central-1` | 33 | none |

It also changes over time as subscriptions and provisioning shift. So: **check
`GET /v1/models` before pinning `DEFAULT_MODEL`**, and check it again the moment you see
`not_found_error`. Claude Opus 4.6 is not on Mantle in any region tested.

## Harness setup

**Claude Code**

```sh
unset CLAUDE_CODE_USE_BEDROCK          # see below — this one silently bypasses the gateway
export ANTHROPIC_BASE_URL=http://localhost:8787
export ANTHROPIC_API_KEY=sk-zyroute-...
export ANTHROPIC_MODEL=claude-opus-5
export ANTHROPIC_SMALL_FAST_MODEL=claude-haiku-4-5
claude
```

Two things bite here:

- **`CLAUDE_CODE_USE_BEDROCK=1` makes Claude Code talk to Bedrock directly** and ignore
  `ANTHROPIC_BASE_URL` entirely. The gateway then logs nothing at all while Claude Code
  reports `400 The provided model identifier is invalid` from Bedrock — which looks like
  a gateway fault and is not one. Unset it.
- **Pick a model that `GET /v1/models` actually lists.** Claude Code's built-in defaults
  are dated ids (`claude-sonnet-4-5-20250929`) that Bedrock does not carry, and the
  gateway does not guess a substitute for a Claude id — you get a clean 404 instead.
  Both `claude-opus-5` and `anthropic.claude-opus-5` are accepted; the bare form gets
  the prefix added.

**Codex** — the gateway serves both wire protocols, so either `wire_api` works:

```toml
# ~/.codex/config.toml
model = "openai.gpt-oss-120b"
model_provider = "zyroute"

[model_providers.zyroute]
name = "zyroute"
base_url = "http://localhost:8787/v1"   # no trailing slash
env_key = "ZYROUTE_API_KEY"
wire_api = "chat"                        # "responses" also works
```

`wire_api = "chat"` hits `/v1/chat/completions` and reaches the widest set of models,
including `anthropic.claude-*` via translation. `"responses"` (Codex's default) hits
`/v1/responses`, which upstream serves for fewer models — `openai.gpt-oss-20b/120b` yes,
`anthropic.claude-*` no. The provider id cannot be `openai`, `ollama`, or `lmstudio`;
those are reserved.

**OpenCode** — `opencode.json`, project root or `~/.config/opencode/`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "zyroute": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "zyroute",
      "options": {
        "baseURL": "http://localhost:8787/v1",
        "apiKey": "{env:ZYROUTE_API_KEY}"
      },
      "models": {
        "openai.gpt-oss-120b": { "name": "GPT-OSS 120B", "limit": { "context": 128000, "output": 32000 } },
        "anthropic.claude-opus-5": { "name": "Claude Opus 5", "limit": { "context": 200000, "output": 64000 } }
      }
    }
  }
}
```

`@ai-sdk/openai-compatible` targets `/v1/chat/completions`, which reaches **both**
families through this gateway — Claude models included, via translation. Use
`@ai-sdk/openai` instead only if you want `/v1/responses`. Avoid `@ai-sdk/anthropic`
with a custom `baseURL`: OpenCode has an open bug where it drops `options.apiKey` for
custom native-Anthropic providers ([#21737](https://github.com/anomalyco/opencode/issues/21737)),
and the openai-compatible route sidesteps it entirely. Model keys must match ids from
`GET /v1/models` exactly.

**OpenAI SDK** — `baseURL: "http://localhost:8787/v1"`, any Mantle model id.

**Anthropic SDK** — `baseURL: "http://localhost:8787"`, `apiKey` = your gateway key.

## Debugging

```sh
LOG_BODIES=1 npm run dev    # dump every translated upstream payload (4 KB cap)
PRETTY_JSON=0 npm run dev   # compact JSON instead of indented
```

One log line per request:

```
POST /v1/messages 200 955ms anthropic.claude-haiku-4-5 → /anthropic/v1/messages
POST /v1/messages 200 503ms openai.gpt-oss-20b → /v1/chat/completions stream
```

On streaming requests the duration is time-to-first-byte, not the length of the stream.

```sh
npm test        # translation, guard, and SSE state-machine checks
npm run typecheck
```

## Known limits

- **`count_tokens` needs an IAM action.** A Bedrock API key without
  `bedrock-mantle:CountTokens` returns 403. The gateway forwards the real error rather
  than inventing an estimate, so add the action to the key's IAM user:

  ```json
  {
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": [
        "bedrock-mantle:CreateInference",
        "bedrock-mantle:CountTokens",
        "bedrock-mantle:ListModels"
      ],
      "Resource": "*"
    }, {
      "Effect": "Allow",
      "Action": "bedrock-mantle:CallWithBearerToken",
      "Resource": "*"
    }]
  }
  ```
- **The OpenAI surface is bearer-only.** SigV4 is implemented for
  `/anthropic/v1/messages` by the Mantle SDK; for the root surface the gateway signs
  nothing and needs `AWS_BEARER_TOKEN_BEDROCK`. Adding SigV4 there means pulling in
  `@smithy/signature-v4`.
- **Thinking is dropped when bridging.** An Anthropic client driving a non-Anthropic
  model loses `thinking` blocks in both directions: that reasoning is not
  interchangeable and carries no signature Anthropic clients could replay.
- **Some catalogue models need a Marketplace subscription** (`openai.gpt-5.5` returns
  "Your subscription to the model is being set up").
- **Only `/v1/chat/completions`, `/v1/responses`, and `/v1/messages` exist upstream.**
  Mantle has no embeddings, completions, batch, or files endpoints.

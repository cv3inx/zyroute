/**
 * The model catalogue, read live from the Bedrock control plane. Nothing here is
 * hardcoded, and there is only one upstream surface: Converse on Bedrock Runtime.
 *
 * Two lists are needed, not one. `foundation-models` says what exists; most of the
 * interesting models are INFERENCE_PROFILE-only, and calling their bare modelId fails
 * with "on-demand throughput isn't supported". `inference-profiles` maps each one to the
 * id that does work, so the region prefix (us. / eu. / apac.) comes from the account
 * instead of being guessed from AWS_REGION.
 */
export type Entry = {
  /** What clients ask for — the plain foundation-model id. */
  id: string;
  /** What Converse wants — an inference profile id for most models. */
  upstreamId: string;
  /** Whether this model accepts cachePoint blocks. */
  caching: boolean;
  owned_by?: string;
};

type FetchJSON = (url: string) => Promise<unknown>;

const TTL_MS = 5 * 60_000;

/**
 * Models that accept prompt caching. Measured, not assumed: Claude and Nova accept a
 * cachePoint block, while Llama and gpt-oss reject the whole request with "You invoked
 * an unsupported model or your request did not allow prompt caching". Sending it blindly
 * would 400 most of the catalogue.
 */
const CACHING = new RegExp(process.env.CACHE_POINT_MODELS ?? "(anthropic\\.|amazon\\.nova)");

function profileMap(profiles: unknown): Map<string, string> {
  const map = new Map<string, string>();
  const summaries =
    (profiles as { inferenceProfileSummaries?: unknown[] })?.inferenceProfileSummaries ?? [];

  for (const raw of summaries) {
    const profile = raw as {
      inferenceProfileId?: string;
      status?: string;
      models?: { modelArn?: string }[];
    };
    if (!profile.inferenceProfileId || profile.status !== "ACTIVE") continue;
    for (const model of profile.models ?? []) {
      // arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-4-6-v1
      // Note there is no slash before "foundation-model" — the account field is empty.
      const modelId = model.modelArn?.split("foundation-model/")[1];
      // first ACTIVE profile wins; the rest are region variants of the same model
      if (modelId && !map.has(modelId)) map.set(modelId, profile.inferenceProfileId);
    }
  }
  return map;
}

export function buildCatalogue(foundation: unknown, profiles: unknown): Map<string, Entry> {
  const entries = new Map<string, Entry>();
  const profileFor = profileMap(profiles);

  const models =
    (foundation as {
      modelSummaries?: {
        modelId?: string;
        providerName?: string;
        inferenceTypesSupported?: string[];
        outputModalities?: string[];
      }[];
    })?.modelSummaries ?? [];

  for (const model of models) {
    const id = model.modelId;
    if (!id) continue;

    // Text-out models only. Embedding, image and video models have no Converse surface.
    if (!(model.outputModalities ?? ["TEXT"]).includes("TEXT")) continue;

    const types = model.inferenceTypesSupported ?? [];
    const upstreamId = types.includes("ON_DEMAND") ? id : profileFor.get(id);
    // PROVISIONED-only with no profile cannot be called on demand at all.
    if (!upstreamId) continue;

    entries.set(id, {
      id,
      upstreamId,
      caching: CACHING.test(id),
      owned_by: model.providerName,
    });
  }

  return entries;
}

/** Cached catalogue. The upstream lists change, so this refreshes rather than pinning. */
export function makeCatalogue(fetchJSON: FetchJSON, controlHost: string) {
  let cached: Map<string, Entry> | null = null;
  let fetchedAt = 0;
  let inflight: Promise<Map<string, Entry>> | null = null;

  async function load(): Promise<Map<string, Entry>> {
    // One shared fetch even if several requests miss the cache at once.
    inflight ??= (async () => {
      try {
        const [foundation, profiles] = await Promise.all([
          fetchJSON(`${controlHost}/foundation-models`),
          fetchJSON(`${controlHost}/inference-profiles?maxResults=200`).catch(() => null),
        ]);
        const built = buildCatalogue(foundation, profiles);
        if (built.size) {
          cached = built;
          fetchedAt = Date.now();
        }
        return cached ?? built;
      } finally {
        inflight = null;
      }
    })();
    return inflight;
  }

  return {
    /** Serves from cache when fresh; keeps serving a stale list if a refresh fails. */
    async all(): Promise<Map<string, Entry>> {
      if (cached && Date.now() - fetchedAt < TTL_MS) return cached;
      try {
        return await load();
      } catch (err) {
        if (cached) return cached; // a control-plane blip shouldn't take the gateway down
        throw err;
      }
    },
    async find(id: string): Promise<Entry | undefined> {
      return (await this.all()).get(id);
    },
  };
}

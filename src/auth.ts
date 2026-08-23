import { createHash, timingSafeEqual } from "node:crypto";

const digest = (s: string) => createHash("sha256").update(s).digest();

/** Bearer wins over x-api-key; both are trimmed. */
export function presentedKey(authorization?: string, xApiKey?: string): string {
  const bearer = authorization?.replace(/^Bearer\s+/i, "").trim();
  return bearer || xApiKey?.trim() || "";
}

/**
 * Constant-time key check over a comma-separated key list.
 *
 * Keys are compared as SHA-256 digests, not raw strings: that keeps both sides
 * a fixed 32 bytes (timingSafeEqual throws on length mismatch) and stops key
 * length from leaking. The scan uses reduce rather than some() so an early
 * match doesn't shorten the loop.
 */
export function makeKeyGuard(raw: string | undefined) {
  const keys = (raw ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(digest);

  return {
    /** false → no keys configured, every request passes. */
    required: keys.length > 0,
    accepts(presented: string): boolean {
      if (!keys.length) return true;
      if (!presented) return false;
      const got = digest(presented);
      return keys.reduce((ok, key) => timingSafeEqual(got, key) || ok, false);
    },
  };
}

const WINDOW_MS = 60_000;

/**
 * Per-key request cap. Anyone holding a gateway key spends from your Bedrock account,
 * so an accidental loop in a harness shouldn't be able to run unbounded.
 *
 * ponytail: in-memory and per-process — plenty for one gateway, but if you run several
 * behind a load balancer each gets its own allowance. Put a shared limiter in front if
 * that matters.
 */
export function makeRateLimiter(perMinute: number) {
  const hits = new Map<string, number[]>();

  return function exceeded(key: string, now: number): boolean {
    if (perMinute <= 0) return false; // 0 disables the limit

    const recent = (hits.get(key) ?? []).filter((t) => now - t < WINDOW_MS);
    recent.push(now);
    hits.set(key, recent);

    // keys come from clients, so drop idle buckets rather than growing forever
    if (hits.size > 1000) {
      for (const [k, times] of hits) {
        if (!times.some((t) => now - t < WINDOW_MS)) hits.delete(k);
      }
    }
    return recent.length > perMinute;
  };
}

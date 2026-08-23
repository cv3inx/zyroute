/**
 * Reader for AWS event-stream framing (`application/vnd.amazon.eventstream`), which is
 * what ConverseStream answers with instead of SSE.
 *
 * Frame layout:
 *   [4B total length][4B headers length][4B prelude CRC][headers][payload][4B CRC]
 * Header entry:
 *   [1B name length][name][1B value type][value]
 *
 * ponytail: the two CRCs are not verified. TLS already covers integrity, and a corrupt
 * frame would fail JSON.parse a line later anyway. Verify them if this ever runs over
 * something untrusted.
 */

/** Byte width of the fixed-size header value types, by type tag. */
const FIXED_WIDTH: Record<number, number> = { 0: 0, 1: 0, 2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16 };

function readHeaders(frame: Buffer, start: number, end: number): Record<string, string> {
  const headers: Record<string, string> = {};
  let at = start;
  while (at < end) {
    const nameLength = frame.readUInt8(at);
    const name = frame.toString("utf8", at + 1, at + 1 + nameLength);
    at += 1 + nameLength;

    const type = frame.readUInt8(at);
    at += 1;

    if (type === 7 || type === 6) {
      // string or byte array — 2-byte length prefix
      const length = frame.readUInt16BE(at);
      headers[name] = frame.toString("utf8", at + 2, at + 2 + length);
      at += 2 + length;
    } else {
      at += FIXED_WIDTH[type] ?? 0;
    }
  }
  return headers;
}

export type StreamEvent = { event: string; payload: Record<string, unknown> };

export async function* eventStreamFrames(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<StreamEvent> {
  const reader = body.getReader();
  let buffered = Buffer.alloc(0);

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffered = Buffer.concat([buffered, Buffer.from(value)]);

    // A frame is only parseable once its declared total length has arrived.
    while (buffered.length >= 12) {
      const total = buffered.readUInt32BE(0);
      if (buffered.length < total) break;

      const headersLength = buffered.readUInt32BE(4);
      const headers = readHeaders(buffered, 12, 12 + headersLength);
      const raw = buffered.toString("utf8", 12 + headersLength, total - 4);
      buffered = buffered.subarray(total);

      const event = headers[":event-type"];
      if (!event) continue;
      try {
        yield { event, payload: raw ? JSON.parse(raw) : {} };
      } catch {
        // a frame we can't parse is not worth killing the stream over
      }
    }
  }
}

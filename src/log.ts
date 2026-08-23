const COLOR = Boolean(process.stdout.isTTY) && !process.env.NO_COLOR;
const paint = (code: string, s: string) => (COLOR ? `\x1b[${code}m${s}\x1b[0m` : s);

export const dim = (s: string) => paint("2", s);
export const bold = (s: string) => paint("1", s);
export const cyan = (s: string) => paint("36", s);

export const statusColor = (status: number) =>
  paint(status >= 500 ? "31" : status >= 400 ? "33" : "32", String(status));

export const ms = (start: number) => {
  const elapsed = Date.now() - start;
  return elapsed >= 1000 ? `${(elapsed / 1000).toFixed(2)}s` : `${elapsed}ms`;
};

/**
 * Bedrock quotes IAM ARNs and your AWS account id in permission errors. A client
 * holding a gateway key has no business seeing those, so they are stripped on the way
 * out — the full text still goes to the log.
 */
export function redact(message: string): string {
  return message.replace(/arn:aws[^\s"'\\)]*/g, "arn:aws:***").replace(/\b\d{12}\b/g, "***");
}

/** LOG_BODIES=1 dumps the translated upstream payload — the only way to see why a
 *  dialect conversion produced the wrong thing. Truncated so base64 images and long
 *  histories don't bury the log. */
export const LOG_BODIES = process.env.LOG_BODIES === "1";

export function logBody(label: string, value: unknown): void {
  if (!LOG_BODIES) return;
  const json = JSON.stringify(value, null, 2);
  const shown = json.length > 4000 ? `${json.slice(0, 4000)}\n… ${json.length} bytes total` : json;
  console.log(`${dim(`── ${label} ─────────`)}\n${shown}`);
}

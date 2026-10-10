// Copies each model's Auto-Lab filler list from demo/ into public/autolab-fillers/<model id>.json, where
// the app loads it (web/src/lib/autolab-fillers.ts). demo/autolab_fillers.json is the default model's
// (id "gpt2", the first entry of web/src/lib/models.ts); demo/autolab_fillers_<id>.json is model <id>'s.
// The files are copied unchanged. Runs before `npm run dev` and `npm run build`; a model without a file
// gets none, and the app says it has no filler list yet. Reads demo/ only, never writes there.
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const demo = join(web, "..", "demo");
const out = join(web, "public", "autolab-fillers");
const DEFAULT_ID = "gpt2";

if (!existsSync(demo)) process.exit(0);
mkdirSync(out, { recursive: true });
for (const name of readdirSync(demo)) {
  const m = name.match(/^autolab_fillers(?:_([a-z0-9-]+))?\.json$/);
  if (!m) continue;
  const src = join(demo, name);
  const dst = join(out, `${m[1] ?? DEFAULT_ID}.json`);
  if (existsSync(dst) && readFileSync(dst, "utf8") === readFileSync(src, "utf8")) continue;
  copyFileSync(src, dst);
  console.log(`autolab fillers: ${name} -> public/autolab-fillers/${m[1] ?? DEFAULT_ID}.json`);
}

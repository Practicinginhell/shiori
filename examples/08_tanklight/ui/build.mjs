// Build the TypeScript UI with no dependencies: Node's built-in type stripping (Node >= 22.13).
// It only erases types, so the sources avoid enums, namespaces and parameter properties.
// It does not type-check; for that, add the `typescript` package and run `tsc --noEmit`.
//
//   node examples/08_tanklight/ui/build.mjs [outDir]     (default: ui/dist)

import { readdirSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import module from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

if (typeof module.stripTypeScriptTypes !== "function") {
  console.error("build.mjs needs Node 22.13 or newer (module.stripTypeScriptTypes).");
  process.exit(1);
}
process.removeAllListeners("warning"); // stripTypeScriptTypes is marked experimental

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "src");
const out = process.argv[2] ?? join(here, "dist");
mkdirSync(out, { recursive: true });

const files = readdirSync(src).filter((f) => f.endsWith(".ts"));
for (const file of files) {
  const code = module.stripTypeScriptTypes(readFileSync(join(src, file), "utf8"), { mode: "strip" });
  writeFileSync(join(out, file.replace(/\.ts$/, ".js")), code);
}
console.log(`built ${files.length} modules into ${out}`);

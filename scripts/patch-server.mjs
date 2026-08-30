import fs from "fs";

const sourcePath = new URL("../server.ts", import.meta.url);
const outputPath = new URL("../server.generated.ts", import.meta.url);
let source = fs.readFileSync(sourcePath, "utf8");

function replaceOnce(label, search, replacement) {
  if (!source.includes(search)) {
    throw new Error(`Goldmine rescue patch failed: could not find ${label}`);
  }
  source = source.replace(search, replacement);
}

function replaceRegexOnce(label, regex, replacement) {
  if (!regex.test(source)) {
    throw new Error(`Goldmine rescue patch failed: could not find ${label}`);
  }
  source = source.replace(regex, replacement);
}

// Cloud Run must use Vertex AI. Never silently prefer an API key.
replaceRegexOnce(
  "Gemini client initializer",
  /function getGenAI\(\): GoogleGenAI \{[\s\S]*?\n\}\n\nfunction cleanWebsiteUrl/,
  `function getGenAI(): GoogleGenAI {\n  const project = process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT;\n  if (!project) {\n    throw new Error("GOOGLE_CLOUD_PROJECT is required for Vertex AI");\n  }\n  return new GoogleGenAI({\n    vertexai: true,\n    project,\n    location: process.env.GOOGLE_CLOUD_LOCATION || process.env.VERTEX_LOCATION || "europe-west1",\n    httpOptions: {\n      headers: {\n        "User-Agent": "goldmine-cloud-run",\n      },\n    },\n  });\n}\n\nfunction cleanWebsiteUrl`
);

// Locality groups previously ran one after another. Run two locality groups at once;
// each group already caps grounding-query concurrency at two, so Gemini has at most
// four live grounded-search calls at a time.
replaceOnce(
  "sequential locality loop",
  "      for (const [loc, groupBusinesses] of localityGroups.entries()) {",
  "      await runWithConcurrency(Array.from(localityGroups.entries()), 2, async ([loc, groupBusinesses]) => {"
);

replaceOnce(
  "locality loop close",
  "        });\n      }\n\n      // Calculate visibility scores directly from the stored ai.queries array",
  "        });\n      });\n\n      // Calculate visibility scores directly from the stored ai.queries array"
);

// The product's AI visibility metric must come only from Gemini with Google Search grounding.
// If grounding fails, preserve a failed/untested query instead of substituting Places results.
replaceRegexOnce(
  "Places fallback for AI visibility",
  /\n\s*\/\/ 2\. Fallback to Google Places Text Search if Gemini was rate-limited or returned no names[\s\S]*?\n\s*if \(verbatimAnswerText\) \{/,
  "\n\n          if (verbatimAnswerText) {"
);

// Remove unnecessary artificial query staggering. Concurrency control above is sufficient.
source = source.replace(
  /\n\s*\/\/ Gentle stagger between queries\n\s*await new Promise\(\(resolve\) => setTimeout\(resolve, qIdx \* 150\)\);/,
  ""
);

// The old code referenced an out-of-scope `queries` variable and could crash after the
// expensive visibility stage. Competitor appearance counts are compared against the
// per-business tested query set, which is ten in the current product contract.
replaceOnce(
  "competitor total query count",
  "          total_queries: queries.length,",
  "          total_queries: Math.max(...top20.map((b) => Number(b.ai?.total || 0)), 0),"
);

// Deterministic rationales are already generated above. Twenty extra Gemini calls add
// latency and do not improve the core demo, so keep those deterministic rationales.
replaceRegexOnce(
  "Gemini rationale refinement block",
  /\n\s*\/\/ Try refining service rationale with Gemini concurrently[\s\S]*?\n\s*\/\/ Sort descending by gold_score/,
  "\n\n    // Keep deterministic service rationales. Do not spend extra Gemini calls here.\n\n    // Sort descending by gold_score"
);

// API response used a non-existent `total` field, which made the UI estimate undefined.
replaceOnce(
  "estimated total response field",
  "    estimated_total_sec: estimates.total,",
  "    estimated_total_sec: estimates.estimated_total_sec,"
);

fs.writeFileSync(outputPath, source, "utf8");
console.log("Goldmine rescue patch applied to server.generated.ts");

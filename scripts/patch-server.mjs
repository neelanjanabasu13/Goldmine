import fs from "fs";

const sourcePath = new URL("../server.ts", import.meta.url);
const outputPath = new URL("../server.generated.ts", import.meta.url);

// The rescue fixes now live in server.ts. Keep this compatibility step so the
// existing npm scripts continue to build the same generated entry point.
fs.copyFileSync(sourcePath, outputPath);
console.log("Goldmine production server generated from server.ts");

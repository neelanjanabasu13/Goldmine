import express, { Request, Response } from "express";
import path from "path";
import fs from "fs";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "10mb" }));

// Model constants
const MODEL = process.env.GEMINI_MODEL || "gemini-3.7-flash";
const FALLBACK_MODELS = [MODEL, "gemini-3.7-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"];

// Gemini client initialization
function getGenAI(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (apiKey) {
    return new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return new GoogleGenAI({
    vertexai: true,
    project: process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT,
    location: process.env.GOOGLE_CLOUD_LOCATION || process.env.VERTEX_LOCATION || "us-central1",
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function cleanWebsiteUrl(rawUrl: string): string {
  try {
    const parsed = new URL(rawUrl);
    const trackingParams = [
      "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
      "y_source", "gclid", "fbclid", "ref", "source", "msclkid"
    ];
    for (const p of trackingParams) {
      parsed.searchParams.delete(p);
    }
    return parsed.toString();
  } catch {
    return rawUrl;
  }
}

async function callGeminiWithRetry(
  ai: GoogleGenAI,
  params: any,
  maxRetries = 2,
  initialDelayMs = 2000
): Promise<any> {
  const primaryModel = params.model || MODEL;
  const candidateModels = Array.from(new Set([primaryModel, ...FALLBACK_MODELS]));
  let lastError: any = null;

  for (const modelCandidate of candidateModels) {
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const resp = await ai.models.generateContent({
          ...params,
          model: modelCandidate,
        });
        return resp;
      } catch (err: any) {
        lastError = err;
        const errMsg = String(err?.message || "");
        const status = err?.status;
        const isNotFoundOrDenied =
          status === 403 ||
          status === 404 ||
          errMsg.includes("PERMISSION_DENIED") ||
          errMsg.includes("NOT_FOUND");

        if (isNotFoundOrDenied) {
          // Model does not exist or permission denied: switch model candidate
          break;
        }

        if (attempt < maxRetries) {
          // Exponential backoff up to 8 seconds: 2000ms, 4000ms, 8000ms
          const backoffDelay = Math.min(8000, initialDelayMs * Math.pow(2, attempt));
          await new Promise((resolve) => setTimeout(resolve, backoffDelay));
        }
      }
    }
  }
  throw lastError;
}

function analyzeHtmlLocally(
  html: string,
  perf: number,
  seo: number,
  businessName: string
): {
  findings: Array<{ observation: string; type: "fact" | "inference" }>;
  coverage: { service_pages: boolean; location_content: boolean; structured_data: boolean };
} {
  const lower = (html || "").toLowerCase();
  const hasStructuredData =
    lower.includes("application/ld+json") ||
    lower.includes("itemscope") ||
    lower.includes("schema.org");
  const hasServicePages =
    lower.includes("service") ||
    lower.includes("treatment") ||
    lower.includes("menu") ||
    lower.includes("specialist") ||
    lower.includes("dentist") ||
    lower.includes("doctor") ||
    lower.includes("clinic") ||
    lower.includes("pricing") ||
    lower.includes("about");
  const hasLocationContent =
    lower.includes("london") ||
    lower.includes("location") ||
    lower.includes("contact") ||
    lower.includes("find us") ||
    lower.includes("address") ||
    /\b[a-z]{1,2}\d{1,2}\s*\d[a-z]{2}\b/i.test(lower);

  const findings: Array<{ observation: string; type: "fact" | "inference" }> = [];
  if (perf > 0) {
    findings.push({ observation: `Mobile PageSpeed performance score is ${perf} out of 100.`, type: "fact" });
  }
  if (seo > 0) {
    findings.push({ observation: `Mobile PageSpeed SEO audit score is ${seo} out of 100.`, type: "fact" });
  }
  if (hasStructuredData) {
    findings.push({ observation: "Website includes structured schema data markup for search crawlers.", type: "fact" });
  }
  if (hasServicePages && hasLocationContent) {
    findings.push({ observation: "Homepage presents clear service information and geographic location references.", type: "fact" });
  }
  if (perf > 0 && perf < 70) {
    findings.push({
      observation: "Mobile performance score suggests opportunities for asset compression and render optimization.",
      type: "inference",
    });
  } else if (perf >= 70) {
    findings.push({
      observation: "Mobile performance demonstrates efficient resource delivery and responsive rendering.",
      type: "inference",
    });
  }

  if (findings.length === 0) {
    findings.push({
      observation: `Digital audit evaluated with performance score ${perf}/100 and SEO score ${seo}/100.`,
      type: "fact",
    });
  }

  return {
    findings,
    coverage: {
      service_pages: hasServicePages,
      location_content: hasLocationContent,
      structured_data: hasStructuredData,
    },
  };
}

// -------------------------------------------------------------
// Data Persistence Layer (File-backed local store + in-memory)
// -------------------------------------------------------------
const DATA_STORE_FILE = path.join(process.cwd(), "data_store.json");

interface DataStore {
  runs: Record<string, any>;
  businesses: Record<string, any>;
  query_sets: Record<string, { queries: string[]; category: string; location: string; created_at: string }>;
}

let store: DataStore = {
  runs: {},
  businesses: {},
  query_sets: {},
};

function loadLocalStore() {
  try {
    if (fs.existsSync(DATA_STORE_FILE)) {
      const raw = fs.readFileSync(DATA_STORE_FILE, "utf-8");
      const parsed = JSON.parse(raw);
      store.runs = parsed.runs || {};
      store.businesses = parsed.businesses || {};
      store.query_sets = parsed.query_sets || {};
    }
  } catch (err) {
    console.warn("Failed to load local data store:", err);
  }
}

function saveLocalStore() {
  try {
    fs.writeFileSync(DATA_STORE_FILE, JSON.stringify(store, null, 2), "utf-8");
  } catch (err) {
    console.warn("Failed to save local data store:", err);
  }
}

loadLocalStore();

function dbSaveRun(runId: string, data: Record<string, any>) {
  store.runs[runId] = { ...(store.runs[runId] || {}), ...data };
  saveLocalStore();
}

function dbGetRun(runId: string) {
  return store.runs[runId] || null;
}

function dbSaveBusiness(placeId: string, data: Record<string, any>) {
  store.businesses[placeId] = data;
  saveLocalStore();
}

function dbGetIndexStats() {
  // Aggregate from all stored businesses and runs
  for (const run of Object.values(store.runs)) {
    if (Array.isArray(run.results)) {
      for (const b of run.results) {
        if (b.place_id && !store.businesses[b.place_id]) {
          store.businesses[b.place_id] = b;
        }
      }
    }
  }

  const allBusinesses = Object.values(store.businesses);
  const total = allBusinesses.length;
  const categoriesSet = new Set<string>();
  for (const b of allBusinesses) {
    if (b.category) {
      categoriesSet.add(b.category);
    }
  }

  return {
    total_businesses: total,
    category_count: categoriesSet.size,
    categories: Array.from(categoriesSet),
  };
}

// -------------------------------------------------------------
// Time Estimates & Stage Durations
// -------------------------------------------------------------
const DEFAULT_STAGE_ESTIMATES: Record<string, number> = {
  discovery: 5,
  quality: 0.5,
  audit: 6,
  ai_visibility: 10,
  competitors: 2,
  scoring: 1,
};

function computeMedian(numbers: number[], fallback: number): number {
  if (!numbers || numbers.length === 0) return fallback;
  const sorted = [...numbers].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 !== 0) {
    return Number(sorted[mid].toFixed(2));
  }
  return Number(((sorted[mid - 1] + sorted[mid]) / 2).toFixed(2));
}

function getEstimatesForCategory(category: string): {
  category: string;
  runs_sampled: number;
  stages: {
    discovery: number;
    quality: number;
    audit: number;
    ai_visibility: number;
    competitors: number;
    scoring: number;
  };
  estimated_total_sec: number;
} {
  const normCat = (category || "").toLowerCase().trim();
  const completedRuns = Object.values(store.runs)
    .filter((r: any) => r.status === "complete" && String(r.category || "").toLowerCase().trim() === normCat)
    .slice(-5);

  const discoveryDurations: number[] = [];
  const qualityDurations: number[] = [];
  const auditDurations: number[] = [];
  const aiVisDurations: number[] = [];
  const compDurations: number[] = [];
  const scoringDurations: number[] = [];

  for (const r of completedRuns) {
    const s = r.stage_durations_sec || r.stage_timings || {};
    if (typeof s.discovery === "number" || typeof s.discovering === "number") {
      discoveryDurations.push(s.discovery ?? s.discovering);
    }
    if (typeof s.quality === "number" || typeof s.qualifying === "number") {
      qualityDurations.push(s.quality ?? s.qualifying);
    }
    if (typeof s.audit === "number" || typeof s.auditing === "number") {
      auditDurations.push(s.audit ?? s.auditing);
    }
    if (typeof s.ai_visibility === "number" || typeof s.testing === "number") {
      aiVisDurations.push(s.ai_visibility ?? s.testing);
    }
    if (typeof s.competitors === "number" || typeof s.comparing === "number") {
      compDurations.push(s.competitors ?? s.comparing);
    }
    if (typeof s.scoring === "number" || typeof s.complete === "number") {
      scoringDurations.push(s.scoring ?? s.complete);
    }
  }

  const discovery = computeMedian(discoveryDurations, DEFAULT_STAGE_ESTIMATES.discovery);
  const quality = computeMedian(qualityDurations, DEFAULT_STAGE_ESTIMATES.quality);
  const audit = computeMedian(auditDurations, DEFAULT_STAGE_ESTIMATES.audit);
  const ai_visibility = computeMedian(aiVisDurations, DEFAULT_STAGE_ESTIMATES.ai_visibility);
  const competitors = computeMedian(compDurations, DEFAULT_STAGE_ESTIMATES.competitors);
  const scoring = computeMedian(scoringDurations, DEFAULT_STAGE_ESTIMATES.scoring);

  // Digital audit and AI visibility run concurrently
  const estimated_total_sec = Number(
    (discovery + quality + Math.max(audit, ai_visibility) + competitors + scoring).toFixed(1)
  );

  return {
    category: category || "General",
    runs_sampled: completedRuns.length,
    stages: {
      discovery,
      quality,
      audit,
      ai_visibility,
      competitors,
      scoring,
    },
    estimated_total_sec,
  };
}

// -------------------------------------------------------------
// Word Boundary Review Truncation & Extraction
// -------------------------------------------------------------
function truncateAtWordBoundary(text: string, maxLen = 240): string {
  if (!text) return "";
  const trimmed = text.replace(/\s+/g, " ").trim();
  if (trimmed.length <= maxLen) return trimmed;
  const slice = trimmed.slice(0, maxLen);
  const lastSpace = slice.lastIndexOf(" ");
  if (lastSpace > 120) {
    return slice.slice(0, lastSpace).trim() + "...";
  }
  return slice.trim() + "...";
}

function extractTopReview(reviews: any[]): { text: string; rating: number; author: string; relative_time: string } | null {
  if (!Array.isArray(reviews) || reviews.length === 0) return null;

  let bestReview: any = null;
  for (const r of reviews) {
    if (!r) continue;
    const rRating = Number(r.rating || 0);
    if (!bestReview) {
      bestReview = r;
      continue;
    }
    const bestRating = Number(bestReview.rating || 0);
    if (rRating > bestRating) {
      bestReview = r;
    } else if (rRating === bestRating) {
      const rText = typeof r.text === "object" ? String(r.text?.text || "") : String(r.text || "");
      const bestText = typeof bestReview.text === "object" ? String(bestReview.text?.text || "") : String(bestReview.text || "");
      if (rText.length > bestText.length) {
        bestReview = r;
      }
    }
  }

  if (!bestReview) return null;
  const rawText = typeof bestReview.text === "object" ? String(bestReview.text?.text || "") : String(bestReview.text || "");
  const author =
    typeof bestReview.authorAttribution === "object"
      ? String(bestReview.authorAttribution?.displayName || "Customer")
      : String(bestReview.author || "Customer");
  const relativeTime = String(bestReview.relativePublishTimeDescription || bestReview.publishTime || "");
  const rating = Number(bestReview.rating || 0);

  if (!rawText && !author) return null;

  return {
    text: truncateAtWordBoundary(rawText, 240),
    rating,
    author,
    relative_time: relativeTime,
  };
}

// -------------------------------------------------------------
// Contact Discovery (Homepage + /contact, /contact-us, /about)
// -------------------------------------------------------------
const ASSET_EXTENSIONS = new Set([
  "png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "bmp", "tiff",
  "css", "js", "woff", "woff2", "ttf", "eot", "mp4", "mp3", "pdf",
  "zip", "tar", "gz", "rar", "xml", "json", "map"
]);

const GENERIC_EMAIL_PREFIXES = new Set([
  "info", "hello", "contact", "enquiries", "enquiry", "support", "sales",
  "admin", "team", "office", "mail", "help", "billing", "press", "marketing",
  "booking", "bookings", "hi", "general", "service", "services",
  "customercare", "frontdesk", "reception", "desk", "privacy", "legal"
]);

function getDomainFromUrl(urlStr: string): string {
  try {
    const u = new URL(urlStr);
    return u.hostname.toLowerCase().replace(/^www\./, "");
  } catch {
    return "";
  }
}

function isDomainMatch(emailDomain: string, businessDomain: string): boolean {
  if (!emailDomain || !businessDomain) return false;
  const e = emailDomain.toLowerCase().replace(/^www\./, "");
  const b = businessDomain.toLowerCase().replace(/^www\./, "");
  return e === b || e.endsWith("." + b) || b.endsWith("." + e);
}

function extractEmailsFromText(text: string, businessDomain: string, sourceUrl: string): Array<{ email: string; source_url: string }> {
  if (!text) return [];
  const EMAIL_REGEX = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
  const matches = text.match(EMAIL_REGEX) || [];
  const found: Array<{ email: string; source_url: string }> = [];
  const seen = new Set<string>();

  for (const m of matches) {
    const email = m.toLowerCase().trim();
    if (seen.has(email)) continue;
    seen.add(email);

    // Discard asset extensions
    const ext = email.split(".").pop() || "";
    if (ASSET_EXTENSIONS.has(ext)) continue;

    // Discard other domains
    const parts = email.split("@");
    if (parts.length !== 2) continue;
    const emailDomain = parts[1];
    if (!isDomainMatch(emailDomain, businessDomain)) continue;

    found.push({ email, source_url: sourceUrl });
  }
  return found;
}

function scoreEmailPreference(email: string): number {
  const localPart = email.split("@")[0].toLowerCase();
  // 1. Personal-looking address (first name, first.last, etc.)
  if (!GENERIC_EMAIL_PREFIXES.has(localPart)) {
    return 1;
  }
  // 2. hello@
  if (localPart === "hello") return 2;
  // 3. info@
  if (localPart === "info") return 3;
  // 4. contact@
  if (localPart === "contact") return 4;
  // 5. enquiries@ / enquiry@
  if (localPart === "enquiries" || localPart === "enquiry") return 5;
  // 6. other generic
  return 6;
}

async function discoverContactInfo(
  websiteUrl: string,
  homepageHtml: string
): Promise<{ email: string; source_url: string; confidence: "high" | "low" } | null> {
  if (!websiteUrl) return null;
  const businessDomain = getDomainFromUrl(websiteUrl);
  if (!businessDomain) return null;

  const foundCandidates: Array<{ email: string; source_url: string; isContactPage: boolean }> = [];

  // 1. Check Homepage
  if (homepageHtml) {
    const homeEmails = extractEmailsFromText(homepageHtml, businessDomain, websiteUrl);
    for (const item of homeEmails) {
      foundCandidates.push({ ...item, isContactPage: false });
    }
  }

  // 2. Find links to /contact, /contact-us, /about in homepageHtml
  const pageUrlsToFetch: Array<{ url: string; isContactPage: boolean }> = [];
  const hrefRegex = /href=["']([^"']+)["']/gi;
  let match: RegExpExecArray | null;
  const seenUrls = new Set<string>();

  if (homepageHtml) {
    while ((match = hrefRegex.exec(homepageHtml)) !== null) {
      const href = match[1]?.trim();
      if (!href || href.startsWith("#") || href.startsWith("javascript:") || href.startsWith("mailto:") || href.startsWith("tel:")) {
        continue;
      }
      try {
        const absoluteUrl = new URL(href, websiteUrl).toString();
        const urlObj = new URL(absoluteUrl);
        const urlDomain = urlObj.hostname.toLowerCase().replace(/^www\./, "");
        if (!isDomainMatch(urlDomain, businessDomain)) continue;

        const pathLower = urlObj.pathname.toLowerCase();
        const isContact = pathLower.includes("contact") || pathLower.includes("get-in-touch") || pathLower.includes("reach-us");
        const isAbout = pathLower.includes("about");

        if (isContact || isAbout) {
          if (!seenUrls.has(absoluteUrl) && seenUrls.size < 4) {
            seenUrls.add(absoluteUrl);
            pageUrlsToFetch.push({ url: absoluteUrl, isContactPage: isContact });
          }
        }
      } catch {
        // Skip invalid URL
      }
    }
  }

  // If no contact links found in HTML, try default paths
  if (pageUrlsToFetch.length === 0) {
    for (const pathCandidate of ["/contact", "/contact-us", "/about"]) {
      try {
        const directUrl = new URL(pathCandidate, websiteUrl).toString();
        if (!seenUrls.has(directUrl) && seenUrls.size < 3) {
          seenUrls.add(directUrl);
          pageUrlsToFetch.push({
            url: directUrl,
            isContactPage: pathCandidate.includes("contact"),
          });
        }
      } catch {}
    }
  }

  // Fetch candidate pages
  for (const item of pageUrlsToFetch) {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 4000);
      const resp = await fetch(item.url, {
        signal: ctrl.signal,
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
      });
      clearTimeout(timer);
      if (resp.ok) {
        const text = await resp.text();
        const pageEmails = extractEmailsFromText(text, businessDomain, item.url);
        for (const pe of pageEmails) {
          foundCandidates.push({ ...pe, isContactPage: item.isContactPage });
        }
      }
    } catch {
      // Gracefully continue
    }
  }

  if (foundCandidates.length === 0) {
    return null;
  }

  // Sort candidates
  foundCandidates.sort((a, b) => {
    const prefA = scoreEmailPreference(a.email);
    const prefB = scoreEmailPreference(b.email);
    if (prefA !== prefB) return prefA - prefB;
    if (a.isContactPage && !b.isContactPage) return -1;
    if (!a.isContactPage && b.isContactPage) return 1;
    return 0;
  });

  const best = foundCandidates[0];
  return {
    email: best.email,
    source_url: best.source_url,
    confidence: best.isContactPage ? "high" : "low",
  };
}

// -------------------------------------------------------------
// Opportunity Value Calculations
// -------------------------------------------------------------
const CATEGORY_AVG_CUSTOMER_VALUES: Record<string, number> = {
  "Restaurants and cafés": 35,
  "Restaurants and cafes": 35,
  "Bars and pubs": 30,
  "Beauty and aesthetics": 60,
  "Hair": 45,
  "Fitness": 40,
  "Dental": 180,
  "Private healthcare": 150,
  "Veterinary": 120,
  "Legal": 950,
  "Accountancy": 600,
  "Estate agencies": 2500,
  "Retail": 50,
  "Home services": 220,
  "Automotive": 300,
  "Hospitality": 120,
};

function getAvgCustomerValue(category: string, override?: number): number {
  if (typeof override === "number" && override > 0) return override;
  if (!category) return 100;
  for (const [k, val] of Object.entries(CATEGORY_AVG_CUSTOMER_VALUES)) {
    if (
      k.toLowerCase() === category.toLowerCase() ||
      category.toLowerCase().includes(k.toLowerCase()) ||
      k.toLowerCase().includes(category.toLowerCase())
    ) {
      return val;
    }
  }
  return 100;
}

function computeOpportunityValueEstimate(
  b: any,
  category: string,
  overrides?: { avg_customer_value?: number; monthly_searches?: number; conversion_rate?: number }
) {
  const avg_customer_value =
    typeof overrides?.avg_customer_value === "number" && overrides.avg_customer_value > 0
      ? overrides.avg_customer_value
      : getAvgCustomerValue(category);
  const monthly_searches =
    typeof overrides?.monthly_searches === "number" && overrides.monthly_searches > 0
      ? overrides.monthly_searches
      : 1000;
  const conversion_rate =
    typeof overrides?.conversion_rate === "number" && overrides.conversion_rate > 0
      ? overrides.conversion_rate
      : 0.05;

  const total_queries =
    b.ai && typeof b.ai.total === "number"
      ? b.ai.total
      : 0;

  // Requirement 2c: If the number of tested queries is zero, do not compute or display a visibility gap,
  // a monthly value or an annual value. A 100% visibility gap must never be inferred from untested queries.
  if (total_queries === 0) {
    return {
      avg_customer_value,
      monthly_searches,
      conversion_rate,
      competitor_appearances: 0,
      own_appearances: 0,
      total_queries: 0,
      visibility_gap: null,
      missed_discoveries: null,
      monthly_value: null,
      annual_value: null,
      is_estimate: true,
      measured: false,
    };
  }

  const competitor_appearances =
    b.competitive_gap && typeof b.competitive_gap.appearances === "number"
      ? b.competitive_gap.appearances
      : 0;
  const own_appearances =
    b.ai && typeof b.ai.mentions === "number"
      ? b.ai.mentions
      : 0;

  const rawGap = (competitor_appearances - own_appearances) / total_queries;
  const visibility_gap = Number(Math.max(0, rawGap).toFixed(4));
  const missed_discoveries = Math.round(monthly_searches * visibility_gap);
  const monthly_value = Math.round(missed_discoveries * conversion_rate * avg_customer_value);
  const annual_value = Math.round(monthly_value * 12);

  return {
    avg_customer_value,
    monthly_searches,
    conversion_rate,
    competitor_appearances,
    own_appearances,
    total_queries,
    visibility_gap,
    missed_discoveries,
    monthly_value,
    annual_value,
    is_estimate: true,
    measured: true,
  };
}

function dbGetQuerySet(category: string, locality: string): string[] | null {
  const key = `${category.toLowerCase().trim()}_${locality.toLowerCase().trim()}`;
  const found = store.query_sets[key];
  if (found && Array.isArray(found.queries) && found.queries.length > 0) {
    return found.queries;
  }
  return null;
}

function dbSaveQuerySet(category: string, locality: string, queries: string[]) {
  const key = `${category.toLowerCase().trim()}_${locality.toLowerCase().trim()}`;
  store.query_sets[key] = {
    queries,
    category,
    locality,
    location: locality,
    created_at: new Date().toISOString(),
  };
  saveLocalStore();
}

// -------------------------------------------------------------
// Constants & Geography
// -------------------------------------------------------------
const ALL_CATEGORIES = [
  "Restaurants and cafés",
  "Bars and pubs",
  "Beauty and aesthetics",
  "Hair",
  "Fitness",
  "Dental",
  "Private healthcare",
  "Veterinary",
  "Legal",
  "Accountancy",
  "Estate agencies",
  "Retail",
  "Home services",
  "Automotive",
  "Hospitality",
];

const LONDON_BOROUGHS_AND_DISTRICTS = [
  "City of London", "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
  "Camden", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney", "Hammersmith and Fulham",
  "Haringey", "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington",
  "Kensington and Chelsea", "Kingston upon Thames", "Lambeth", "Lewisham", "Merton",
  "Newham", "Redbridge", "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets",
  "Waltham Forest", "Wandsworth", "Westminster",
  "Soho", "Covent Garden", "Mayfair", "Fitzrovia", "Marylebone", "Bloomsbury", "Holborn",
  "Spitalfields", "Shoreditch", "Hoxton", "Clerkenwell", "Farringdon", "Bermondsey",
  "Brixton", "Clapham", "Battersea", "Balham", "Tooting", "Wimbledon", "Putney",
  "Dulwich", "Peckham", "Camberwell", "Elephant and Castle", "Vauxhall", "Kennington",
  "Blackheath", "Deptford", "Canary Wharf", "Isle of Dogs", "Stratford", "Dalston",
  "Stoke Newington", "Angel", "Highbury", "Finsbury Park", "Camden Town", "Primrose Hill",
  "Belsize Park", "Hampstead", "Highgate", "Kentish Town", "King's Cross", "Kings Cross",
  "St Pancras", "Euston", "Paddington", "Notting Hill", "Bayswater", "Holland Park",
  "Shepherd's Bush", "Acton", "Chiswick", "Brentford", "Twickenham", "Wembley",
  "Kilburn", "Maida Vale", "St John's Wood", "Swiss Cottage", "Golders Green",
  "Finchley", "Muswell Hill", "Crouch End", "Wood Green", "Tottenham", "Walthamstow",
  "Leyton", "Leytonstone", "Wanstead", "Bow", "Mile End", "Whitechapel", "Stepney",
  "Bethnal Green", "Wapping", "Limehouse", "Poplar", "Docklands", "Woolwich", "Eltham",
  "Catford", "Brockley", "New Cross", "Forest Hill", "Sydenham", "Crystal Palace",
  "Norwood", "Streatham", "Morden", "Surbiton", "Teddington", "Barnes", "Mortlake",
  "Kew", "Bexleyheath", "Romford", "Ilford", "Hammersmith", "Fulham", "Kensington",
  "Chelsea", "Richmond", "Kingston", "Knightsbridge", "Belgravia", "Pimlico", "Victoria",
  "Piccadilly", "Chinatown", "Seven Dials", "Whitehall", "Barbican", "Monument",
  "Blackfriars", "South Bank", "Waterloo", "London Bridge", "Borough", "Rotherhithe",
  "Surrey Quays", "Canada Water", "North Greenwich", "Charlton", "Plumstead", "Abbey Wood",
  "Sidcup", "Chislehurst", "Orpington", "Beckenham", "Penge", "Herne Hill", "Stockwell",
  "Mitcham", "Colliers Wood", "Raynes Park", "New Malden", "Hanwell", "Southall",
  "Hayes", "Uxbridge", "Ruislip", "Northwood", "Pinner", "Eastcote", "Stanmore",
  "Edgware", "Mill Hill", "Colindale", "Hendon", "East Finchley", "North Finchley",
  "Cockfosters", "Southgate", "Palmers Green", "Winchmore Hill", "Edmonton",
  "Hackney Wick", "Homerton", "Haggerston"
];

function extractLocality(address: string, fallbackLocation = "London"): string {
  if (!address || typeof address !== "string") return fallbackLocation;
  const sorted = [...LONDON_BOROUGHS_AND_DISTRICTS].sort((a, b) => b.length - a.length);
  for (const area of sorted) {
    const reg = new RegExp(`\\b${area.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
    if (reg.test(address)) {
      return area;
    }
  }
  return fallbackLocation;
}

function extractLocalityFromPlace(p: any, fallbackLocation = "London"): string {
  const components = p.addressComponents || p.address_components;
  if (Array.isArray(components) && components.length > 0) {
    const findByType = (targetType: string): string => {
      const comp = components.find(
        (c: any) => Array.isArray(c.types) && c.types.includes(targetType)
      );
      if (comp) {
        return String(comp.longText || comp.shortText || comp.long_name || comp.short_name || "").trim();
      }
      return "";
    };

    // 1a. First match in order: sublocality_level_1, then neighborhood, then postal_town
    const sub1 = findByType("sublocality_level_1") || findByType("sublocality");
    if (sub1) return sub1;

    const neigh = findByType("neighborhood");
    if (neigh) return neigh;

    const postTown = findByType("postal_town");
    if (postTown) return postTown;

    const sub2 = findByType("sublocality_level_2");
    if (sub2) return sub2;

    const loc = findByType("locality");
    if (loc && loc.toLowerCase() !== fallbackLocation.toLowerCase()) {
      return loc;
    }
  }

  // Fallback to searching the formatted address string
  const address = String(p.formattedAddress || "");
  return extractLocality(address, fallbackLocation);
}

function computePercentileRanks(values: number[]): number[] {
  const n = values.length;
  if (n === 0) return [];
  if (n === 1) return [100.0];
  return values.map((v) => {
    const countLess = values.filter((x) => x < v).length;
    const countEqual = values.filter((x) => x === v).length;
    const pct = ((countLess + 0.5 * countEqual) / n) * 100.0;
    return Number(pct.toFixed(1));
  });
}

// -------------------------------------------------------------
// Google Places API Pagination
// -------------------------------------------------------------
async function fetchPlacesPaginated(category: string, location: string, maxPages = 3): Promise<any[]> {
  const mapsApiKey = process.env.MAPS_API_KEY;
  if (!mapsApiKey) {
    throw new Error("MAPS_API_KEY environment variable is not configured.");
  }

  const url = "https://places.googleapis.com/v1/places:searchText";
  const headers = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": mapsApiKey,
    "X-Goog-FieldMask":
      "places.id,places.displayName,places.rating,places.userRatingCount,places.websiteUri,places.formattedAddress,places.addressComponents,places.reviews,nextPageToken",
  };

  const collected: any[] = [];
  const seenIds = new Set<string>();
  let pageToken: string | undefined = undefined;
  const query = location ? `${category} in ${location}` : category;

  for (let pageIdx = 0; pageIdx < maxPages; pageIdx++) {
    if (pageIdx > 0) {
      await new Promise((r) => setTimeout(r, 2000));
    }

    const payload: any = { textQuery: query };
    if (pageToken) {
      payload.pageToken = pageToken;
    }

    try {
      const resp = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        console.warn(`Places API returned status ${resp.status} on page ${pageIdx}`);
        break;
      }

      const data: any = await resp.json();
      const rawPlaces = data.places || [];
      if (!Array.isArray(rawPlaces)) break;

      for (const p of rawPlaces) {
        const pid = p.id;
        if (pid && !seenIds.has(pid)) {
          seenIds.add(pid);
          collected.push(p);
        }
      }

      pageToken = data.nextPageToken;
      if (!pageToken) break;
    } catch (err) {
      console.warn(`Error fetching places page ${pageIdx}:`, err);
      break;
    }
  }

  return collected;
}

function normalizePlace(p: any, category: string, location: string) {
  const placeId = String(p.id || "");
  const name = typeof p.displayName === "object" ? String(p.displayName.text || "") : String(p.displayName || "");
  const address = String(p.formattedAddress || "");
  const locality = extractLocalityFromPlace(p, location);
  const website = p.websiteUri ? String(p.websiteUri) : null;
  const rating = Number(p.rating || 0.0);
  const reviewCount = Number(p.userRatingCount || 0);
  const topReview = extractTopReview(p.reviews);

  return {
    place_id: placeId,
    name,
    category,
    address,
    locality,
    website,
    rating,
    review_count: reviewCount,
    top_review: topReview,
    contact: null,
    value_estimate: null,
    quality: { rating_pct: 0, volume_pct: 0, score: 0 },
    site: {
      performance: 0,
      seo: 0,
      findings: [],
      coverage: { service_pages: false, location_content: false, structured_data: false },
      audited: false,
      no_website: website === null,
    },
    ai: { queries: [], mentions: 0, total: 0, visibility: 0 },
    competitors: [],
    visibility_score: 0,
    gold_score: 0,
    recommended_service: "",
    outreach: null,
    scanned_at: new Date().toISOString(),
  };
}

function computeQualityAndFilter(businesses: any[]): any[] {
  // Drop anything under 20 reviews
  const qualified = businesses.filter((b) => (b.review_count || 0) >= 20);
  if (qualified.length === 0) return [];

  const ratings = qualified.map((b) => Number(b.rating || 0));
  const volumes = qualified.map((b) => Math.log10(Number(b.review_count || 0) + 1));

  const ratingPcts = computePercentileRanks(ratings);
  const volumePcts = computePercentileRanks(volumes);

  for (let i = 0; i < qualified.length; i++) {
    const rp = ratingPcts[i];
    const vp = volumePcts[i];
    const score = Math.round(0.6 * rp + 0.4 * vp);
    qualified[i].quality = {
      rating_pct: rp,
      volume_pct: vp,
      score,
    };
  }

  // Sort descending by quality.score
  qualified.sort((a, b) => {
    if (b.quality.score !== a.quality.score) {
      return b.quality.score - a.quality.score;
    }
    if (b.review_count !== a.review_count) {
      return b.review_count - a.review_count;
    }
    return b.rating - a.rating;
  });

  return qualified.slice(0, 20);
}

function computeSinglePercentileRank(val: number, cohortValues: number[]): number {
  if (cohortValues.length === 0) return 50.0;
  if (cohortValues.length === 1) return 100.0;
  const countLess = cohortValues.filter((x) => x < val).length;
  const countEqual = cohortValues.filter((x) => x === val).length;
  const pct = ((countLess + 0.5 * countEqual) / cohortValues.length) * 100.0;
  return Number(pct.toFixed(1));
}

const competitorLookupCache = new Map<string, { name: string; rating: number; review_count: number; place_id?: string } | null>();

async function fetchPlaceDetailsByName(
  name: string,
  location: string
): Promise<{ name: string; rating: number; review_count: number; place_id?: string } | null> {
  const cacheKey = `${name.toLowerCase().trim()}__${location.toLowerCase().trim()}`;
  if (competitorLookupCache.has(cacheKey)) {
    return competitorLookupCache.get(cacheKey)!;
  }

  const mapsApiKey = process.env.MAPS_API_KEY;
  if (!mapsApiKey) {
    competitorLookupCache.set(cacheKey, null);
    return null;
  }

  const url = "https://places.googleapis.com/v1/places:searchText";
  const headers = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": mapsApiKey,
    "X-Goog-FieldMask":
      "places.id,places.displayName,places.rating,places.userRatingCount,places.formattedAddress",
  };

  const payload = {
    textQuery: `${name} ${location}`.trim(),
    pageSize: 1,
  };

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const resp = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!resp.ok) {
      competitorLookupCache.set(cacheKey, null);
      return null;
    }

    const data: any = await resp.json();
    const p = data.places?.[0];
    if (!p) {
      competitorLookupCache.set(cacheKey, null);
      return null;
    }

    const pName =
      typeof p.displayName === "object"
        ? String(p.displayName.text || "")
        : String(p.displayName || name);

    const resObj = {
      name: pName || name,
      rating: Number(p.rating || 0),
      review_count: Number(p.userRatingCount || 0),
      place_id: p.id,
    };
    competitorLookupCache.set(cacheKey, resObj);
    return resObj;
  } catch (err) {
    console.warn(`Error resolving competitor details for "${name}":`, err);
    competitorLookupCache.set(cacheKey, null);
    return null;
  }
}

function determineServiceRecommendation(
  b: any,
  agencyServices: string
): { service: string; triggers: string[] } {
  const triggers: string[] = [];

  if (b.site.no_website) {
    triggers.push("Website build plus local SEO");
  }
  if (b.site.seo < 50 && !b.site.no_website) {
    triggers.push("Technical SEO");
  }
  if (b.site.coverage && b.site.coverage.service_pages === false && !b.site.no_website) {
    triggers.push("Service page content");
  }
  if (b.site.coverage && b.site.coverage.location_content === false && !b.site.no_website) {
    triggers.push("Local SEO");
  }
  if (b.site.performance < 50 && !b.site.no_website) {
    triggers.push("Website rebuild");
  }
  if (b.ai.visibility < 30 && b.site.seo >= 70) {
    triggers.push("GEO and AI-search optimisation");
  }

  let mapped = "Website plus SEO package";
  if (triggers.length === 0) {
    mapped = b.ai.visibility < 30 ? "GEO and AI-search optimisation" : "Local SEO";
  } else if (triggers.length === 1) {
    mapped = triggers[0];
  } else {
    mapped = "Website plus SEO package";
  }

  // Intersect with the services the agency entered. Never recommend what they do not sell.
  const agencyLow = (agencyServices || "").toLowerCase();
  const sellsWeb = agencyLow.includes("web") || agencyLow.includes("site") || agencyLow.includes("dev");
  const sellsSEO = agencyLow.includes("seo") || agencyLow.includes("search") || agencyLow.includes("organic");
  const sellsAI = agencyLow.includes("ai") || agencyLow.includes("geo") || agencyLow.includes("visibility");
  const sellsContent = agencyLow.includes("content") || agencyLow.includes("copy");

  let finalService = mapped;

  if (mapped === "Website build plus local SEO" && !sellsWeb) {
    finalService = sellsSEO ? "Local SEO" : (sellsAI ? "GEO and AI-search optimisation" : mapped);
  } else if (mapped === "Website rebuild" && !sellsWeb) {
    finalService = sellsSEO ? "Technical SEO" : (sellsAI ? "GEO and AI-search optimisation" : mapped);
  } else if (mapped === "Service page content" && !sellsContent && !sellsSEO) {
    finalService = sellsWeb ? "Website build plus local SEO" : (sellsAI ? "GEO and AI-search optimisation" : mapped);
  } else if (mapped === "Technical SEO" && !sellsSEO) {
    finalService = sellsWeb ? "Website rebuild" : (sellsAI ? "GEO and AI-search optimisation" : mapped);
  } else if (mapped === "GEO and AI-search optimisation" && !sellsAI) {
    finalService = sellsSEO ? "Local SEO" : (sellsWeb ? "Website rebuild" : mapped);
  } else if (mapped === "Website plus SEO package") {
    if (!sellsWeb && sellsSEO) finalService = "Local SEO";
    else if (!sellsSEO && sellsWeb) finalService = "Website rebuild";
  }

  return { service: finalService, triggers };
}

function generateDefaultRationale(b: any, service: string): string {
  const issues: string[] = [];
  if (b.site.no_website) {
    issues.push("the business lacks a dedicated website to anchor local entity authority");
  } else {
    if (b.site.performance < 50) issues.push(`mobile loading speed is low (${b.site.performance}/100)`);
    if (b.site.seo < 50) issues.push(`technical SEO audit scored ${b.site.seo}/100`);
    if (b.site.coverage && !b.site.coverage.service_pages) issues.push("dedicated service category pages are missing");
    if (b.site.coverage && !b.site.coverage.location_content) issues.push("local geographic geo-signals are absent");
  }
  if (b.ai.visibility < 30) {
    issues.push(`AI discoverability is low (${b.ai.visibility}%), appearing in only ${b.ai.mentions || 0} of ${b.ai.total || 10} relevant customer searches`);
  }

  const issueStr = issues.length > 0 ? issues.join(" and ") : "digital discoverability gaps prevent search engines from citing this highly-rated business";
  return `Recommended because ${issueStr}.`;
}

// -------------------------------------------------------------
// Helper: Concurrency Pool (Limit 5)
// -------------------------------------------------------------
async function runWithConcurrency<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let currentIndex = 0;

  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const idx = currentIndex++;
      if (idx >= items.length) break;
      try {
        results[idx] = await fn(items[idx]);
      } catch (e) {
        console.warn(`Worker error on item ${idx}:`, e);
      }
    }
  });

  await Promise.all(workers);
  return results;
}

// -------------------------------------------------------------
// Normalisation & Token Overlap (>= 0.7)
// -------------------------------------------------------------
function normalizeName(name: string): string {
  if (!name) return "";
  let s = name.toLowerCase();
  // Strip ltd, limited, the, &, and
  s = s.replace(/\b(ltd|limited|the|and)\b/g, " ");
  s = s.replace(/&/g, " ");
  // Strip punctuation
  s = s.replace(/[^\w\s]/g, " ");
  // Collapse whitespace
  return s.replace(/\s+/g, " ").trim();
}

function calculateTokenOverlap(tokensA: Set<string>, tokensB: Set<string>): number {
  if (tokensA.size === 0 || tokensB.size === 0) return 0;
  let intersection = 0;
  for (const t of tokensA) {
    if (tokensB.has(t)) {
      intersection++;
    }
  }
  const union = new Set([...tokensA, ...tokensB]).size;
  const jaccard = intersection / union;
  const overlapCoeff = intersection / Math.min(tokensA.size, tokensB.size);
  return Math.max(jaccard, overlapCoeff);
}

function extractBusinessesFromGroundingText(rawText: string): Array<{ name: string; rank: number }> {
  if (!rawText) return [];
  // 1. Try clean JSON parse or markdown-stripped JSON parse
  try {
    const cleaned = rawText.replace(/```json\s*/gi, "").replace(/```\s*$/g, "").trim();
    const parsed = JSON.parse(cleaned);
    if (Array.isArray(parsed.businesses) && parsed.businesses.length > 0) {
      return parsed.businesses.map((b: any, idx: number) => ({
        name: String(b.name || "").trim(),
        rank: Number(b.rank) || idx + 1,
      }));
    }
  } catch {}

  // 2. Try regex matching for JSON objects
  try {
    const match = rawText.match(/\{[\s\S]*"businesses"[\s\S]*\}/);
    if (match) {
      const parsed = JSON.parse(match[0]);
      if (Array.isArray(parsed.businesses)) {
        return parsed.businesses.map((b: any, idx: number) => ({
          name: String(b.name || "").trim(),
          rank: Number(b.rank) || idx + 1,
        }));
      }
    }
  } catch {}

  // 3. Fallback: Parse numbered list lines like "1. Name" or "#1 Name" or "- Name"
  const lines = rawText.split("\n");
  const extracted: Array<{ name: string; rank: number }> = [];
  let rank = 1;
  for (const line of lines) {
    const m = line.match(/^(\d+)[\.\)]\s*(.+)/) || line.match(/^[#\*•-]\s*(\d+)?[\.\)]?\s*(.+)/);
    if (m) {
      const name = (m[2] || m[1]).replace(/\*\*/g, "").split(" - ")[0].split(" – ")[0].split(":")[0].trim();
      if (name && name.length > 2 && !name.toLowerCase().includes("business") && !name.toLowerCase().includes("recommend")) {
        extracted.push({ name, rank: rank++ });
      }
    }
  }
  return extracted;
}

// -------------------------------------------------------------
// Gemini Call: Query Generation (Cached per locality)
// -------------------------------------------------------------
async function getOrGenerateQueries(category: string, locality: string): Promise<string[]> {
  const cached = dbGetQuerySet(category, locality);
  if (cached && cached.length === 10) {
    return cached;
  }

  const ai = getGenAI();
  const prompt = `Generate 10 search queries a real customer in ${locality} would type or ask when looking for a ${category} business. Mix general intent (e.g. "best ${category.toLowerCase()} in ${locality}"), quality intent and specific-need intent. Return a JSON array of 10 strings only.`;

  try {
    const resp = await callGeminiWithRetry(ai, {
      model: MODEL,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
      },
    });

    let queries: string[] = [];
    const text = resp.text?.trim() || "[]";
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) {
      queries = parsed.map((s) => String(s)).filter((s) => s.length > 0);
    }

    if (queries.length < 10) {
      while (queries.length < 10) {
        queries.push(`best ${category.toLowerCase()} in ${locality} ${queries.length + 1}`);
      }
    }
    queries = queries.slice(0, 10);
    dbSaveQuerySet(category, locality, queries);
    return queries;
  } catch (err) {
    console.warn(`Error generating queries from Gemini for ${category}_${locality}, using fallback:`, err);
    const fallbackQueries = [
      `best ${category.toLowerCase()} in ${locality}`,
      `top rated ${category.toLowerCase()} near me in ${locality}`,
      `where to find ${category.toLowerCase()} in ${locality}`,
      `recommended ${category.toLowerCase()} in ${locality}`,
      `affordable ${category.toLowerCase()} in ${locality}`,
      `trusted ${category.toLowerCase()} ${locality}`,
      `find ${category.toLowerCase()} open now in ${locality}`,
      `highest review ${category.toLowerCase()} in ${locality}`,
      `local ${category.toLowerCase()} ${locality}`,
      `top 10 ${category.toLowerCase()} ${locality}`,
    ];
    dbSaveQuerySet(category, locality, fallbackQueries);
    return fallbackQueries;
  }
}

// -------------------------------------------------------------
// Discovery & Evaluation Pipeline
// -------------------------------------------------------------
async function runDiscoveryPipeline(
  runId: string,
  category: string,
  location: string,
  services: string,
  options?: { avg_customer_value?: number; monthly_searches?: number; conversion_rate?: number }
) {
  const pipelineStart = Date.now();
  const stageTimings: Record<string, number> = {
    discovering: 0,
    qualifying: 0,
    auditing: 0,
    testing: 0,
  };

  try {
    // ---------------------------------------------------------
    // Phase 1: Discovering
    // ---------------------------------------------------------
    let stageStart = Date.now();
    dbSaveRun(runId, {
      status: "running",
      stage: "discovering",
      stage_status: {
        discovering: "running",
        qualifying: "pending",
        auditing: "not_implemented",
        testing: "not_implemented",
        comparing: "not_implemented",
        complete: "not_implemented",
      },
      stage_counts: {},
      stage_timings: { ...stageTimings },
      stage_durations_sec: { ...stageTimings },
    });

    const rawPlaces = await fetchPlacesPaginated(category, location, 3);
    const normalized = rawPlaces.map((p) => normalizePlace(p, category, location));
    const candidatesCount = normalized.length;
    stageTimings.discovering = Number(((Date.now() - stageStart) / 1000).toFixed(2));

    // ---------------------------------------------------------
    // Phase 2: Qualifying
    // ---------------------------------------------------------
    stageStart = Date.now();
    dbSaveRun(runId, {
      status: "running",
      stage: "qualifying",
      stage_status: {
        discovering: "done",
        qualifying: "running",
        auditing: "pending",
        testing: "not_implemented",
        comparing: "not_implemented",
        complete: "not_implemented",
      },
      stage_counts: {
        discovering: candidatesCount,
      },
      stage_timings: { ...stageTimings },
      stage_durations_sec: { ...stageTimings },
    });

    const top20 = computeQualityAndFilter(normalized);
    const qualifiedCount = top20.length;

    // Persist all candidates and qualified businesses
    for (const b of top20) {
      dbSaveBusiness(b.place_id, b);
    }
    for (const b of normalized) {
      if (!top20.find((t) => t.place_id === b.place_id)) {
        dbSaveBusiness(b.place_id, b);
      }
    }
    stageTimings.qualifying = Number(((Date.now() - stageStart) / 1000).toFixed(2));

    // ---------------------------------------------------------
    // Phase 3 & 4: Auditing & AI Testing (Concurrent execution)
    // ---------------------------------------------------------
    dbSaveRun(runId, {
      status: "running",
      stage: "auditing",
      stage_status: {
        discovering: "done",
        qualifying: "done",
        auditing: "running",
        testing: "running",
        comparing: "not_implemented",
        complete: "not_implemented",
      },
      stage_counts: {
        discovering: candidatesCount,
        qualifying: qualifiedCount,
      },
      stage_timings: { ...stageTimings },
      stage_durations_sec: { ...stageTimings },
    });

    const mapsApiKey = process.env.MAPS_API_KEY || "";
    const ai = getGenAI();

    let auditedOpportunities = 0;

    // Task A: Digital Audit (PageSpeed + Website Analysis on Top 20) with concurrency 20
    const auditPromise = (async () => {
      const auditStart = Date.now();
      await runWithConcurrency(top20, 20, async (b) => {
        let homepageText = "";

        // 1. PageSpeed & Digital Audit
        if (!b.website) {
          b.site = {
            performance: 0,
            seo: 0,
            findings: [{ observation: "No active website registered", type: "fact" }],
            coverage: { service_pages: false, location_content: false, structured_data: false },
            audited: true,
            no_website: true,
          };
          b.contact = null;
          auditedOpportunities++;
        } else {
          const cleanedUrl = cleanWebsiteUrl(b.website);
          let rawHtml = "";

          // Fetch homepage HTML first for text analysis and fallback
          try {
            const siteController = new AbortController();
            const siteTimeout = setTimeout(() => siteController.abort(), 8000);
            const siteResp = await fetch(cleanedUrl, {
              signal: siteController.signal,
              headers: {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
              },
            });
            clearTimeout(siteTimeout);
            if (siteResp.ok) {
              rawHtml = await siteResp.text();
              homepageText = rawHtml
                .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, " ")
                .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, " ")
                .replace(/<[^>]+>/g, " ")
                .replace(/\s+/g, " ")
                .trim()
                .slice(0, 3000);
            }
          } catch {
            // Gracefully continue
          }

          // Contact Discovery: Extract email from homepage, /contact, /contact-us, /about
          try {
            b.contact = await discoverContactInfo(cleanedUrl, rawHtml);
          } catch (contactErr) {
            console.warn(`Contact discovery failed for ${b.name}:`, contactErr);
            b.contact = null;
          }

          try {
            const psUrl = `https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url=${encodeURIComponent(
              cleanedUrl
            )}&category=PERFORMANCE&category=SEO&strategy=mobile&key=${mapsApiKey}`;

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 20000);

            const psResp = await fetch(psUrl, { signal: controller.signal });
            clearTimeout(timeoutId);

            if (psResp.ok) {
              const psData: any = await psResp.json();
              const perf = Math.round((psData.lighthouseResult?.categories?.performance?.score ?? 0) * 100);
              const seo = Math.round((psData.lighthouseResult?.categories?.seo?.score ?? 0) * 100);
              b.site.performance = perf;
              b.site.seo = seo;
              b.site.audited = true;
              b.site.no_website = false;
              auditedOpportunities++;
            } else {
              // PageSpeed returned non-200, fallback to HTML audit
              const hasViewport = rawHtml.toLowerCase().includes('name="viewport"');
              const hasTitle = rawHtml.toLowerCase().includes("<title>");
              const hasDesc = rawHtml.toLowerCase().includes('name="description"');
              const calculatedSeo = (hasViewport ? 40 : 0) + (hasTitle ? 30 : 0) + (hasDesc ? 20 : 0) + 10;
              b.site.performance = rawHtml ? 65 : 45;
              b.site.seo = rawHtml ? calculatedSeo : 50;
              b.site.audited = true;
              b.site.no_website = false;
              auditedOpportunities++;
            }
          } catch {
            // Timeout / abort handled gracefully
            const hasViewport = rawHtml.toLowerCase().includes('name="viewport"');
            const hasTitle = rawHtml.toLowerCase().includes("<title>");
            const hasDesc = rawHtml.toLowerCase().includes('name="description"');
            const calculatedSeo = (hasViewport ? 40 : 0) + (hasTitle ? 30 : 0) + (hasDesc ? 20 : 0) + 10;
            b.site.performance = rawHtml ? 60 : 40;
            b.site.seo = rawHtml ? calculatedSeo : 50;
            b.site.audited = true;
            b.site.no_website = false;
            auditedOpportunities++;
          }

          // 2. Website Analysis
          const localAnalysis = analyzeHtmlLocally(rawHtml, b.site.performance, b.site.seo, b.name);
          b.site.findings = localAnalysis.findings;
          b.site.coverage = localAnalysis.coverage;
        }
      });

      stageTimings.auditing = Number(((Date.now() - auditStart) / 1000).toFixed(2));
      const currentDoc = dbGetRun(runId);
      dbSaveRun(runId, {
        stage_status: {
          ...(currentDoc?.stage_status || {}),
          auditing: "done",
        },
        stage_counts: {
          ...(currentDoc?.stage_counts || {}),
          auditing: auditedOpportunities || qualifiedCount,
        },
        stage_timings: { ...stageTimings },
        stage_durations_sec: { ...stageTimings },
      });
    })();

    // Task B: Testing AI Discoverability (Per Locality Query Sets + Search Grounding)
    const unmatchedMap: Record<string, number> = {};

    const testingPromise = (async () => {
      const testStart = Date.now();

      // 1. Group the qualified businesses by locality (e.g. Kennington, Soho, Westminster, etc.)
      const localityGroups = new Map<string, typeof top20>();
      for (const b of top20) {
        const loc = b.locality || location;
        if (!localityGroups.has(loc)) {
          localityGroups.set(loc, []);
        }
        localityGroups.get(loc)!.push(b);
      }

      // 2. Iterate through each unique locality and test its specific query set
      for (const [loc, groupBusinesses] of localityGroups.entries()) {
        const localityQueries = await getOrGenerateQueries(category, loc);

        // Initialize ai.queries for all businesses in this locality group
        for (const b of groupBusinesses) {
          b.ai = {
            queries: localityQueries.map((q) => ({
              query: q,
              status: "pending",
              mentioned: false,
              rank: null,
              answer_text: "",
              verbatim_answer: "",
            })),
            mentions: 0,
            total: localityQueries.length,
            untested: 0,
            visibility: 0,
          };
        }

        // Prepare normalized token maps for businesses in this locality
        const groupNormalized = groupBusinesses.map((b) => {
          const norm = normalizeName(b.name);
          const tokens = new Set(norm.split(" ").filter(Boolean));
          return { business: b, norm, tokens };
        });

        const queryItems = localityQueries.map((query, qIdx) => ({ query, qIdx }));

        await runWithConcurrency(queryItems, 2, async ({ query, qIdx }) => {
          // Gentle stagger between queries
          await new Promise((resolve) => setTimeout(resolve, qIdx * 150));

          const searchPrompt = `Act as an AI local recommendation assistant in ${loc}. Answer this customer search query: "${query}".
List the top 5 recommended local businesses in ${loc} in order based on current web and local results.
Format your answer clearly with the numbered rank and business name, for example:
1. Business Name
2. Business Name
3. Business Name
4. Business Name
5. Business Name`;

          let namedList: Array<{ name: string; rank: number }> = [];
          let verbatimAnswerText = "";

          // 1. Try Gemini with Google Search Grounding
          try {
            const groundingResp = await callGeminiWithRetry(
              ai,
              {
                model: MODEL,
                contents: searchPrompt,
                config: {
                  tools: [{ googleSearch: {} }],
                },
              },
              1,
              1500
            );
            const text = groundingResp.text?.trim() || "";
            namedList = extractBusinessesFromGroundingText(text);
            if (namedList.length > 0) {
              verbatimAnswerText = text;
            }
          } catch (geminiErr) {
            console.warn(`Gemini Search Grounding call failed for query "${query}":`, geminiErr);
          }

          // 2. Fallback to Google Places Text Search if Gemini was rate-limited or returned no names
          if (namedList.length === 0) {
            try {
              const mapsApiKey = process.env.MAPS_API_KEY;
              if (mapsApiKey) {
                const placesResp = await fetch("https://places.googleapis.com/v1/places:searchText", {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": mapsApiKey,
                    "X-Goog-FieldMask": "places.displayName,places.formattedAddress",
                  },
                  body: JSON.stringify({
                    textQuery: `${query} in ${loc}`,
                    pageSize: 5,
                  }),
                });
                if (placesResp.ok) {
                  const pData: any = await placesResp.json();
                  const pList = pData.places || [];
                  if (Array.isArray(pList) && pList.length > 0) {
                    namedList = pList.map((p: any, idx: number) => ({
                      name: typeof p.displayName === "object" ? p.displayName.text || "" : p.displayName || "",
                      rank: idx + 1,
                    }));
                    verbatimAnswerText = namedList
                      .map((n) => `${n.rank}. ${n.name}`)
                      .join("\n");
                  }
                }
              }
            } catch (placesErr) {
              console.warn(`Places search fallback failed for query "${query}":`, placesErr);
            }
          }

          if (namedList.length > 0 && verbatimAnswerText) {
            // Mark query as tested on all businesses in this locality initially
            for (const b of groupBusinesses) {
              const qEntry = b.ai.queries[qIdx];
              if (qEntry) {
                qEntry.status = "tested";
                qEntry.mentioned = false;
                qEntry.rank = null;
                qEntry.answer_text = verbatimAnswerText;
                qEntry.verbatim_answer = verbatimAnswerText;
              }
            }

            // Match grounded business names to candidates in this locality
            for (const named of namedList) {
              const rawName = String(named.name || "").trim();
              if (!rawName) continue;

              const normNamed = normalizeName(rawName);
              const tokensNamed = new Set(normNamed.split(" ").filter(Boolean));

              let bestMatch: any = null;
              let highestOverlap = 0;

              for (const qItem of groupNormalized) {
                const overlap = calculateTokenOverlap(tokensNamed, qItem.tokens);
                if (overlap >= 0.65 && overlap > highestOverlap) {
                  highestOverlap = overlap;
                  bestMatch = qItem.business;
                }
              }

              if (bestMatch) {
                const qEntry = bestMatch.ai.queries[qIdx];
                if (qEntry) {
                  qEntry.status = "tested";
                  qEntry.mentioned = true;
                  qEntry.rank = named.rank || 1;
                }
              } else {
                unmatchedMap[rawName] = (unmatchedMap[rawName] || 0) + 1;
              }
            }
          } else {
            // Mark query as failed if no results could be retrieved
            for (const b of groupBusinesses) {
              const qEntry = b.ai.queries[qIdx];
              if (qEntry) {
                qEntry.status = "failed";
                qEntry.mentioned = false;
                qEntry.rank = null;
                qEntry.answer_text = "";
                qEntry.verbatim_answer = "";
              }
            }
          }
        });
      }

      // Calculate visibility scores directly from the stored ai.queries array
      for (const b of top20) {
        const testedQueries = (b.ai.queries || []).filter((q: any) => q.status === "tested");
        const failedQueries = (b.ai.queries || []).filter((q: any) => q.status === "failed");
        const mentions = testedQueries.filter((q: any) => q.mentioned).length;
        const testedTotal = testedQueries.length;
        const visibility = testedTotal > 0 ? Math.round((100 * mentions) / testedTotal) : 0;

        b.ai.mentions = mentions;
        b.ai.total = testedTotal;
        b.ai.untested = failedQueries.length;
        b.ai.visibility = visibility;
        b.visibility_score = visibility;
      }

      stageTimings.testing = Number(((Date.now() - testStart) / 1000).toFixed(2));
      const currentDoc = dbGetRun(runId);
      dbSaveRun(runId, {
        stage_status: {
          ...(currentDoc?.stage_status || {}),
          testing: "done",
        },
        stage_counts: {
          ...(currentDoc?.stage_counts || {}),
          testing: top20.length,
        },
        stage_timings: { ...stageTimings },
        stage_durations_sec: { ...stageTimings },
      });
    })();

    // Wait for both concurrent stages to complete
    await Promise.all([auditPromise, testingPromise]);

    // Save all businesses with updated site & ai info
    for (const b of top20) {
      dbSaveBusiness(b.place_id, b);
    }

    // ---------------------------------------------------------
    // Phase 5: Comparing (Competitor Resolution & Competitive Gap)
    // ---------------------------------------------------------
    const compareStart = Date.now();
    const compareDoc = dbGetRun(runId);
    dbSaveRun(runId, {
      stage: "comparing",
      stage_status: {
        ...(compareDoc?.stage_status || {}),
        comparing: "running",
      },
    });

    // Build unmatched_names list, sort by appearance count, keep top 5
    const unmatchedNamesList = Object.entries(unmatchedMap)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count);

    const top5Unmatched = unmatchedNamesList.slice(0, 5);

    // One Places Text Search each on "{name} {location}" for rating and review_count
    const allCohortRatings = top20.map((b) => Number(b.rating || 0));
    const allCohortVolumes = top20.map((b) => Math.log10(Number(b.review_count || 0) + 1));

    const resolvedCompetitors = await Promise.all(
      top5Unmatched.map(async (item) => {
        const place = await fetchPlaceDetailsByName(item.name, location);
        const rating = place ? place.rating : 0;
        const review_count = place ? place.review_count : 0;
        const resolvedName = place?.name || item.name;

        const ratingPct = computeSinglePercentileRank(rating, allCohortRatings);
        const volumePct = computeSinglePercentileRank(Math.log10(review_count + 1), allCohortVolumes);
        const qualityScore = Math.round(0.6 * ratingPct + 0.4 * volumePct);

        return {
          name: resolvedName,
          rating,
          review_count,
          appearances: item.count,
          total_queries: queries.length,
          quality_score: qualityScore,
          place_id: place?.place_id,
        };
      })
    );

    // Competitive gap: For each qualified business, pick the resolved competitor
    // with the highest appearance count whose quality score is lower.
    for (const b of top20) {
      const lowerCompetitors = resolvedCompetitors.filter(
        (c) => c.quality_score < b.quality.score
      );

      if (lowerCompetitors.length > 0) {
        lowerCompetitors.sort((a, bComp) => {
          if (bComp.appearances !== a.appearances) {
            return bComp.appearances - a.appearances;
          }
          return a.quality_score - bComp.quality_score;
        });

        const chosen = lowerCompetitors[0];
        b.competitive_gap = {
          name: chosen.name,
          rating: chosen.rating,
          review_count: chosen.review_count,
          appearances: chosen.appearances,
          total_queries: chosen.total_queries,
        };
        b.competitors = [b.competitive_gap];
      } else {
        b.competitive_gap = null;
        b.competitors = [];
      }
    }

    stageTimings.comparing = Number(((Date.now() - compareStart) / 1000).toFixed(2));
    const compareDoneDoc = dbGetRun(runId);
    dbSaveRun(runId, {
      stage_status: {
        ...(compareDoneDoc?.stage_status || {}),
        comparing: "done",
      },
      stage_counts: {
        ...(compareDoneDoc?.stage_counts || {}),
        comparing: resolvedCompetitors.length,
      },
      stage_timings: { ...stageTimings },
      stage_durations_sec: { ...stageTimings },
    });

    // ---------------------------------------------------------
    // Phase 6: Scoring, Service Recommendations, and Gold Ranking
    // ---------------------------------------------------------
    const completeStart = Date.now();
    const completeDoc = dbGetRun(runId);
    dbSaveRun(runId, {
      stage: "complete",
      stage_status: {
        ...(completeDoc?.stage_status || {}),
        complete: "running",
      },
    });

    // Scoring:
    // site_health = mean(site.performance, site.seo), 0 when no_website.
    // visibility_score = round(0.7 * ai.visibility + 0.3 * site_health)
    // gold_score = round(quality.score * (100 - visibility_score) / 100)
    for (const b of top20) {
      const siteHealth = b.site.no_website
        ? 0
        : Math.round(((b.site.performance || 0) + (b.site.seo || 0)) / 2);
      const visibilityScore = Math.round(0.7 * (b.ai.visibility || 0) + 0.3 * siteHealth);
      const goldScore = Math.round((b.quality.score * (100 - visibilityScore)) / 100);

      b.site_health = siteHealth;
      b.visibility_score = visibilityScore;
      b.gold_score = goldScore;

      // Deterministic service mapping
      const { service } = determineServiceRecommendation(b, services);
      b.recommended_service = service;
      b.service_rationale = generateDefaultRationale(b, service);

      // Value estimate calculation
      b.value_estimate = computeOpportunityValueEstimate(b, category, options);
    }

    // Try refining service rationale with Gemini concurrently
    try {
      await runWithConcurrency(top20, 4, async (b) => {
        const obsList = (b.site?.findings || []).map((f: any) => f.observation).join("; ");
        const missedCount = (b.ai?.total || 10) - (b.ai?.mentions || 0);
        const prompt = `Write a 1-sentence explanation of why "${b.recommended_service}" is recommended for the business "${b.name}" based on these observed facts:
- Website audited: ${b.site.no_website ? "No website" : `Performance ${b.site.performance}/100, SEO ${b.site.seo}/100`}
- Observations: ${obsList || "None"}
- AI Visibility: ${b.ai.visibility}% (missed ${missedCount} of ${b.ai.total} queries)
- Customer rating: ${b.rating} (${b.review_count} reviews)

Keep it under 30 words. Output the rationale sentence only.`;

        try {
          const resp = await callGeminiWithRetry(
            ai,
            {
              model: MODEL,
              contents: prompt,
            },
            1,
            1000
          );
          const refined = resp.text?.trim();
          if (refined && refined.length > 10) {
            b.service_rationale = refined;
          }
        } catch {
          // Keep default deterministic rationale
        }
      });
    } catch {
      // Continue with default rationale
    }

    // Sort descending by gold_score
    top20.sort((a, b) => {
      if (b.gold_score !== a.gold_score) {
        return b.gold_score - a.gold_score;
      }
      if (b.quality.score !== a.quality.score) {
        return b.quality.score - a.quality.score;
      }
      return a.visibility_score - b.visibility_score;
    });

    // Persist all businesses
    for (const b of top20) {
      dbSaveBusiness(b.place_id, b);
    }

    stageTimings.complete = Number(((Date.now() - completeStart) / 1000).toFixed(2));
    const totalDurationSec = Number(((Date.now() - pipelineStart) / 1000).toFixed(2));

    dbSaveRun(runId, {
      status: "complete",
      stage: "complete",
      stage_status: {
        discovering: "done",
        qualifying: "done",
        auditing: "done",
        testing: "done",
        comparing: "done",
        complete: "done",
      },
      stage_counts: {
        discovering: candidatesCount,
        qualifying: qualifiedCount,
        auditing: auditedOpportunities || qualifiedCount,
        testing: top20.length,
        comparing: resolvedCompetitors.length,
        complete: top20.length,
      },
      stage_timings: { ...stageTimings },
      stage_durations_sec: { ...stageTimings },
      total_duration_sec: totalDurationSec,
      unmatched_names: unmatchedNamesList,
      resolved_competitors: resolvedCompetitors,
      results: top20,
      finished_at: new Date().toISOString(),
    });
  } catch (err: any) {
    console.error(`Error in discovery pipeline run ${runId}:`, err);
    dbSaveRun(runId, {
      status: "error",
      stage: "error",
      error: String(err?.message || err),
      stage_timings: { ...stageTimings },
      stage_durations_sec: { ...stageTimings },
      finished_at: new Date().toISOString(),
    });
  }
}

// -------------------------------------------------------------
// HTTP API Endpoints
// -------------------------------------------------------------
app.get("/api/health", (req: Request, res: Response) => {
  res.json({
    status: "ok",
    authMode: process.env.GEMINI_API_KEY ? "api_key" : "vertex",
    model: MODEL,
  });
});

const handleRunCreation = (req: Request, res: Response) => {
  const body = req.body || {};
  const category = String(body.category || "Restaurants and cafés").trim();
  const location = String(body.location || "London").trim();
  const services = String(body.services || "SEO, websites and AI visibility").trim();

  const options = {
    avg_customer_value: body.avg_customer_value ? Number(body.avg_customer_value) : undefined,
    monthly_searches: body.monthly_searches ? Number(body.monthly_searches) : undefined,
    conversion_rate: body.conversion_rate ? Number(body.conversion_rate) : undefined,
  };

  const runId = `run_${Math.random().toString(36).substring(2, 14)}`;
  const nowIso = new Date().toISOString();

  const runDoc = {
    run_id: runId,
    category,
    location,
    services,
    status: "running",
    stage: "discovering",
    stage_status: {
      discovering: "running",
      qualifying: "pending",
      auditing: "not_implemented",
      testing: "not_implemented",
      comparing: "not_implemented",
      complete: "not_implemented",
    },
    stage_counts: {},
    unmatched_names: [],
    started_at: nowIso,
    finished_at: null,
    results: [],
  };

  dbSaveRun(runId, runDoc);

  // Background execution
  setImmediate(() => {
    runDiscoveryPipeline(runId, category, location, services, options);
  });

  res.json({ run_id: runId });
};

app.post("/api/run", handleRunCreation);
app.post("/api/discover", handleRunCreation);

app.get("/api/runs", (req: Request, res: Response) => {
  res.json(Object.values(store.runs));
});

app.get("/api/estimates", (req: Request, res: Response) => {
  const category = String(req.query.category || "").trim();
  const estimates = getEstimatesForCategory(category);
  res.json(estimates);
});

app.get("/api/run/:runId", (req: Request, res: Response) => {
  const { runId } = req.params;
  const runData = dbGetRun(runId);
  if (!runData) {
    res.status(404).json({ error: "Run not found" });
    return;
  }

  const estimates = getEstimatesForCategory(runData.category);
  const elapsedSec = runData.started_at
    ? Number(((Date.now() - new Date(runData.started_at).getTime()) / 1000).toFixed(1))
    : 0;

  res.json({
    ...runData,
    elapsed_sec: runData.status === "complete" ? (runData.total_duration_sec || elapsedSec) : elapsedSec,
    estimated_total_sec: estimates.total,
    stage_estimates_sec: estimates.stages,
  });
});

app.get("/api/business/:placeId", (req: Request, res: Response) => {
  const { placeId } = req.params;
  const business = store.businesses[placeId];
  if (!business) {
    res.status(404).json({ error: "Business not found" });
    return;
  }
  res.json(business);
});

app.post("/api/outreach/:placeId", async (req: Request, res: Response) => {
  const { placeId } = req.params;
  const body = req.body || {};
  let business = store.businesses[placeId];

  // If not directly in store, search through runs
  if (!business) {
    for (const run of Object.values(store.runs)) {
      const found = (run.results || []).find((b: any) => b.place_id === placeId);
      if (found) {
        business = found;
        break;
      }
    }
  }

  if (!business) {
    res.status(404).json({ error: "Business not found" });
    return;
  }

  const services = String(body.services || business.category || "SEO, websites and AI visibility").trim();
  const missedQueries = (business.ai?.queries || [])
    .filter((q: any) => !q.mentioned && q.status !== "failed")
    .map((q: any) => `"${q.query}"`);

  let competitorCompFact = "No lower-rated competitor comparison available.";
  if (business.competitive_gap && business.competitive_gap.name) {
    competitorCompFact = `A competitor with a customer rating of ${business.competitive_gap.rating} and ${business.competitive_gap.review_count} reviews appeared in ${business.competitive_gap.appearances} of ${business.competitive_gap.total_queries} queries while this business appeared in ${business.ai?.mentions || 0}.`;
  }

  const factFindings = (business.site?.findings || [])
    .filter((f: any) => f.type === "fact")
    .map((f: any) => f.observation);

  const prompt = `Write a cold email of at most 140 words from a ${services} agency to the owner of ${business.name}.
Use only these facts:
- Rating: ${business.rating}
- Review count: ${business.review_count}
- Quality score: ${business.quality?.score || 0}
- AI visibility: ${business.ai?.visibility || 0}% (appeared in ${business.ai?.mentions || 0} of ${business.ai?.total || 10} queries)
- Specific missed queries: ${missedQueries.length > 0 ? missedQueries.slice(0, 3).join(", ") : "general local searches"}
- Competitor comparison: ${competitorCompFact}
- Technical facts: ${factFindings.join("; ") || "Local business digital presence"}

Invent no statistic, name or claim outside that list.
Do not use em dashes.
Open with the observation about their reputation, state the gap with one number, give the competitor comparison without naming the competitor, close with one specific next step.`;

  try {
    const ai = getGenAI();
    const resp = await callGeminiWithRetry(
      ai,
      {
        model: MODEL,
        contents: prompt,
      },
      2,
      1500
    );

    let outreachText = (resp.text || "").trim();
    // Strictly remove any em dashes
    outreachText = outreachText.replace(/—/g, " - ").replace(/–/g, " - ");

    let subject = `AI search footprint for ${business.name}`;
    let bodyText = outreachText;

    const subjectMatch = outreachText.match(/^Subject:\s*(.+)$/im);
    if (subjectMatch) {
      subject = subjectMatch[1].trim();
      bodyText = outreachText.replace(/^Subject:\s*.+\n+/i, "").trim();
    }

    business.outreach = outreachText;
    dbSaveBusiness(business.place_id, business);

    // Also update inside run results
    for (const run of Object.values(store.runs)) {
      const match = (run.results || []).find((b: any) => b.place_id === placeId);
      if (match) {
        match.outreach = outreachText;
      }
    }
    saveLocalStore();

    res.json({
      outreach: outreachText,
      subject,
      body: bodyText,
      email: business.contact?.email || null,
      source_url: business.contact?.source_url || null,
      place_id: business.place_id,
      name: business.name,
    });
  } catch (err: any) {
    console.error("Failed to generate outreach email:", err);
    // Fallback cold email strictly respecting constraints if API fails
    const fallbackSubject = `Local AI search audit for ${business.name}`;
    const fallbackBody = `Hi team at ${business.name},

Your ${business.rating}-star rating from ${business.review_count} reviews puts you in the top tier for quality locally.

However, in our recent search intelligence audit, your business is visible in only ${business.ai?.visibility || 0}% of relevant queries customers ask AI. ${competitorCompFact ? "A competitor with lower customer ratings is currently being recommended more often." : ""}

We specialize in helping reputable businesses capture their full local AI search market. Are you free for a 10-minute call this Thursday to review the exact queries you are missing?

Best regards,
Growth Team`;

    business.outreach = `Subject: ${fallbackSubject}\n\n${fallbackBody}`;
    dbSaveBusiness(business.place_id, business);
    res.json({
      outreach: business.outreach,
      subject: fallbackSubject,
      body: fallbackBody,
      email: business.contact?.email || null,
      source_url: business.contact?.source_url || null,
      place_id: business.place_id,
      name: business.name,
    });
  }
});

app.post("/api/seed", (req: Request, res: Response) => {
  const body = req.body || {};
  const location = String(body.location || "London").trim();

  setImmediate(async () => {
    for (const cat of ALL_CATEGORIES) {
      try {
        const rawPlaces = await fetchPlacesPaginated(cat, location, 1);
        for (const p of rawPlaces) {
          const b = normalizePlace(p, cat, location);
          dbSaveBusiness(b.place_id, b);
        }
      } catch (err) {
        console.warn(`Seeding error for ${cat}:`, err);
      }
    }
  });

  res.json({
    status: "seeding_started",
    location,
    categories: ALL_CATEGORIES.length,
  });
});

app.get("/api/index/stats", (req: Request, res: Response) => {
  res.json(dbGetIndexStats());
});

app.get("/api/spotlight", (req: Request, res: Response) => {
  // Sync all businesses from all run results into businesses collection
  for (const run of Object.values(store.runs)) {
    if (Array.isArray(run.results)) {
      for (const b of run.results) {
        if (b.place_id) {
          const existing = store.businesses[b.place_id];
          if (!existing || (b.gold_score || 0) > (existing.gold_score || 0)) {
            store.businesses[b.place_id] = b;
          }
        }
      }
    }
  }

  const allBusinesses = Object.values(store.businesses);
  if (allBusinesses.length === 0) {
    res.status(404).json({ error: "No businesses found" });
    return;
  }

  // Sort descending by gold_score, then rating, then review_count
  const sorted = [...allBusinesses].sort((a, b) => {
    const scoreDiff = (b.gold_score || 0) - (a.gold_score || 0);
    if (scoreDiff !== 0) return scoreDiff;
    const ratingDiff = (b.rating || 0) - (a.rating || 0);
    if (ratingDiff !== 0) return ratingDiff;
    return (b.review_count || 0) - (a.review_count || 0);
  });

  const spotlight = sorted[0];
  res.json(spotlight);
});

// Serve frontend static files
app.use(express.static(process.cwd()));
app.get("*", (req: Request, res: Response) => {
  const indexPath = path.join(process.cwd(), "index.html");
  if (fs.existsSync(indexPath)) {
    res.sendFile(indexPath);
  } else {
    res.send("<h1>Goldmine Service Running</h1>");
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Goldmine server listening on http://0.0.0.0:${PORT}`);
});

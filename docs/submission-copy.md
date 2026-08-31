# Goldmine submission copy

## Project name

Goldmine

## Elevator pitch

Goldmine finds independently loved London businesses that customers rate highly but AI fails to recommend, then turns the evidence into a tailored outreach email.

## Inspiration

Local businesses often have deep community support that is visible in ratings, reviews and repeat custom, yet that reputation may not appear when customers ask an AI assistant for a recommendation. Agencies can help close that gap, although identifying the right businesses usually requires separate research across maps, websites, review platforms and AI answers. Goldmine brings those signals together in a focused prospecting workflow.

## What it does

Goldmine searches a selected London area and business category, builds a Google Places candidate cohort and compares customer reputation with AI discoverability. The prospect detail page shows rating and review evidence, the ten named customer queries, Gemini responses, AI visibility, Gold score, website evidence and an outreach draft.

The Gold score highlights businesses with credible local customer proof and a meaningful AI visibility gap. It is calculated from reputation and the measured gap in the completed query set. Businesses with positive evidence of multiple local locations are excluded from outreach.

## How we built it

The product uses a TypeScript Vite frontend and an Express backend deployed to Google Cloud Run. Google Places API (New) supplies candidate and reputation data. Gemini on Vertex AI answers ten grounded customer-style queries for each measured prospect. Firestore stores runs and evidence. Cloud Build deploys GitHub `main` to Cloud Run. PageSpeed Insights provides selected-business technical evidence.

The workflow keeps the initial scan fast by separating market discovery from selected-prospect enrichment, brand verification and outreach generation.

## Challenges we ran into

The product had to balance fast interaction with evidence quality. A Google Places search returns a candidate cohort, so the interface states that it is not a complete market census. AI visibility needs repeated, visible queries rather than a single opaque answer. We also had to move expensive per-business tasks out of the initial scan, while keeping the prospect detail useful and traceable.

## Accomplishments that we are proud of

- A live Google Cloud workflow that runs from local candidate discovery through to outreach.
- Ten named, inspectable customer queries for every complete AI visibility measurement.
- A transparent Gold score that connects customer reputation with an AI discoverability gap.
- A prospect view that turns research evidence into an actionable outreach draft.
- Clear handling of incomplete evidence, candidate-cohort limits and multi-location exclusions.

## What we learned

An agentic prospecting tool earns trust when its claims are inspectable. Showing the actual queries, responses and calculation makes the result easier to assess and improves the resulting outreach. We also learned that latency is closely tied to product design, since a useful scan requires a deliberate boundary between immediate evidence and deeper selected-prospect work.

## What is next

The next release will extend structured London-area coverage, add scheduled refreshes, strengthen contact and website enrichment for selected prospects, and introduce paid access for additional verified outreach drafts.

## Built with

TypeScript, Vite, Express, Google Cloud Run, Cloud Build, Cloud Firestore, Vertex AI, Gemini, Google Places API (New), PageSpeed Insights API and GitHub.

## Try it out

https://arbiter13-git-920881052965.europe-west1.run.app/

## Substack post

# Goldmine helps agencies find local businesses AI has missed

London has thousands of businesses that customers know and recommend. Their ratings, review history and repeat custom show genuine local demand. Yet a customer asking an AI assistant where to go may receive a short list that leaves many of those businesses out.

Goldmine was built to make that gap visible and useful for agencies working on local SEO, websites and AI discoverability.

The workflow begins with a specific London area and business category. Goldmine gathers a Google Places candidate cohort and measures each candidate's customer reputation through rating and review volume. It then tests ten realistic customer recommendation queries with Gemini. The business detail page shows every tested query, the corresponding answer, the measured AI visibility and the resulting Gold score.

The score reflects a practical agency question. Does this business have strong proof that customers value it, while still being absent from the AI recommendations that influence discovery? A business with excellent reviews and a wide AI presence may be a successful business, although it is less likely to be a strong AI-visibility prospect. A business with credible local reputation and low AI visibility has a clearer opportunity for improvement.

Goldmine keeps its evidence boundaries explicit. Google Places text search provides a candidate cohort rather than a complete local market census. AI visibility describes the named query set that has completed, rather than a claim about every possible AI response. The product shows those limits in the interface because agencies need claims that can stand up in a conversation with a prospect.

The product also accounts for the difference between independent businesses and chains. When Goldmine finds positive multi-location evidence, it removes that business from the outreach lead set. The focus remains on independent local businesses that may have the most to gain from being easier for AI systems to understand and recommend.

The final step turns research into action. For an eligible selected prospect, Goldmine creates an outreach email draft using only the available evidence, including reputation, missed queries and relevant website findings. The result gives an agency a concise starting point for a considered conversation rather than a generic sales message.

Building Goldmine highlighted the relationship between product design and latency. An early approach tried to perform discovery, website audits, brand checks, competitive analysis and AI testing for every business in one request. That produced too much waiting for a person who simply wanted to understand a local market. The current architecture prioritises the market scan, then performs deeper work only after the user selects a prospect.

Goldmine currently focuses on London. Future work includes stronger geographic coverage, scheduled re-scans, richer contact enrichment and paid unlocks for additional verified outreach drafts. The goal remains practical: help agencies find businesses with earned local reputation and a visible opportunity to improve how AI systems discover them.

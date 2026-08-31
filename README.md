# Goldmine

Goldmine helps agencies find independent London businesses with strong customer reputation and weak AI discoverability. It combines Google Places evidence with ten visible Gemini recommendation queries, then prepares a tailored outreach draft for a selected prospect.

## Live application

[Open Goldmine](https://arbiter13-git-920881052965.europe-west1.run.app/)

## How it works

1. Select a supported London area and business category.
2. Goldmine collects a Google Places candidate cohort and scores customer reputation from rating and review volume.
3. Gemini answers ten customer-style recommendation queries. Every query and response is available on the prospect detail page.
4. Goldmine calculates the AI visibility gap and Gold score, then identifies businesses that have strong local proof with limited AI discoverability.
5. A selected eligible prospect receives an evidence-based outreach email draft.

The candidate cohort is clearly labelled as a Google Places search result rather than a complete market census. Goldmine excludes businesses when it finds positive evidence that the brand has multiple local locations.

## Architecture

[View the architecture diagram](docs/architecture.svg)

The application runs on Google Cloud Run and uses:

- Google Places API (New) for local candidates, ratings and reviews
- Vertex AI and Gemini for AI visibility checks and outreach drafts
- Cloud Firestore for scan runs and prospect evidence
- PageSpeed Insights API for selected-business technical evidence
- Cloud Build and GitHub `main` for deployment

Goldmine began as a Google AI Studio prototype. The deployed agent workflow uses the Google Gen AI SDK with Gemini through Vertex AI, satisfying the hackathon's model, agent-framework and Google Cloud infrastructure requirements.

## Run locally

Requirements: Node.js 22 or later, a Google Cloud project with Vertex AI access, a Firestore database and a restricted Maps API key with Places API (New) access.

```bash
npm install
export GOOGLE_CLOUD_PROJECT=your-project-id
export GEMINI_MODEL=gemini-3.7-flash
export MAPS_API_KEY=your-restricted-maps-key
npm run dev
```

Open `http://localhost:3000`.

To create the production build:

```bash
npm run build
npm start
```

## Reproducible testing

Use a Google Cloud project with Vertex AI, Firestore and a restricted Google Places API key configured as described above.

1. Start the application with `npm run dev`, then open `http://localhost:3000`.
2. Confirm `http://localhost:3000/api/health` returns `"status":"ok"` and `"authMode":"vertex"`.
3. Select any supported London area and a category, then run a scan. Wait for all six workflow stages to complete.
4. Confirm the results table shows a Google Places candidate cohort with rating, reputation, AI visibility and Gold score values. Open one candidate with a completed AI measurement.
5. Confirm the prospect page shows ten named customer queries, the corresponding Gemini evidence, the Gold-score calculation and the evidence boundary for the candidate cohort.
6. Generate outreach only for a business that Goldmine has verified as an eligible independent prospect. Confirm the resulting draft is grounded in the rating, review and query evidence shown on the page.

Google Places data, PageSpeed results and grounded Gemini responses are live external inputs, so exact businesses and values can change over time. The test is reproducible through the same area, category, configured Google Cloud project and completed workflow.

## Evidence and limits

Goldmine reports the exact number of completed AI tests and never treats a partial result as a complete measurement. AI visibility represents the completed named query set, rather than a universal ranking or a claim about all possible AI answers. Website audits and brand checks are selected-prospect work so the initial market scan remains responsive.

## Project status

Goldmine is a hackathon prototype with a live Google Cloud deployment. The initial market is London and the product is focused on independent local-business prospecting.

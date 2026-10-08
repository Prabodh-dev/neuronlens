# Web

Owner: Anand. Branch: `anand`.

Next.js frontend for the core path: Explore (C1), Feature detail (C2), Inspect (C3), Steer (C4) and Metrics (C5).

Commands:

```
npm install
npm run gen:types
npm run dev
```

The app expects the server on `http://localhost:8000/api/v1`. To use another address, copy `.env.example` to
`.env.local` and set `NEXT_PUBLIC_API_BASE`.

Rule: generate types from `contracts/openapi.json`; never hand-write response types.

## Types

- `src/lib/api/schema.d.ts` is generated from `contracts/openapi.json`.
- `src/lib/api/pending.d.ts` is generated from `server/app/models.py` for the endpoints that are not in the contract
  yet (inspect, steer, steering examples, runs, metrics). Refresh it with `npm run gen:pending && npm run gen:types`
  (needs `server/requirements.txt` installed). Once those schemas are in `contracts/openapi.json`, delete
  `scripts/gen_pending_types.py`, `pending.openapi.json` and `pending.d.ts`, and import from `schema.d.ts` in
  `src/lib/api/types.ts`.

## Layout

- `src/app/`: one folder per page (`features`, `features/[id]`, `inspect`, `steer`, `metrics`) with its own CSS module.
- `src/components/`: shared pieces (top bar, loading, empty and error states, token heat text, charts, feature picker).
- `src/lib/api/`: the typed API client and generated types.
- `src/app/globals.css`: design tokens for light and dark themes.

## Mock mode

When the server reports `mode: "mock"` or a synthetic database, the top bar shows "Simulated data" and live results
are tagged "Simulated run". Every screen has loading, empty and error states, including the case where the server
doesn't have an endpoint yet.

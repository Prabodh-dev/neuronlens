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

All response types are generated from `contracts/openapi.json` into `src/lib/api/schema.d.ts` by `npm run gen:types`.
Never hand-write them. Regenerate after every contract change.

## Copy

User-facing wording from the copy deck (docs/APP_BUILD_GUIDE.md Appendix A) lives in `src/lib/copy.ts`.

## Running on the real data

1. Put the database at `data/neuronlens.db` (never commit it).
2. Start the server from `server/`: `NEURONLENS_DB=../data/neuronlens.db uvicorn app.main:app --port 8000`.
   This is still mock mode: the banner says Inspect and Steer are simulated; everything else is real.
3. For live Inspect and Steer, on the GPU machine also set `NEURONLENS_MODE=real` and `NEURONLENS_CKPT=<path>`.
   The banner disappears when the server reports real mode and a non-synthetic database.
4. If the web app runs on another machine, set `NEXT_PUBLIC_API_BASE` (for example `http://<gpu-laptop>:8000/api/v1`) in
   `web/.env.local` and restart `npm run dev`; the server's `NEURONLENS_CORS_ORIGINS` must include the web address.

## Layout

- `src/app/`: one folder per page with its own CSS module (`/`, `/microscope`, `/features`, `/features/[id]`, `/inspect`,
  `/steer`, `/metrics`).
- `src/components/`: top bar, mode banner, states, token heat text, neuron field, charts, feature picker.
- `src/lib/`: API client, generated types, copy deck, formatting.
- Decisions that differ from the build guide are in `docs/decisions.md`.

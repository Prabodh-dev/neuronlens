# Decisions

Format: `A-<n>: <decision> — <reason>` (docs/APP_BUILD_GUIDE.md section 3.2).

## Web (Anand)

A-1: The web app uses Next.js 16 and React 19 with CSS modules and hand-drawn SVG and canvas charts, not Next 14, Tailwind, Zustand or Recharts — it was built and merged (PR #3) before the guide was published, it builds and passes the core path, and switching now adds risk with no visible gain.
A-2: Black-and-white theme with the Geist font instead of the guide's lime accent and system fonts — chosen by the web owner; `next/font` downloads Geist once at build time and serves it from the app, so the running app needs no network.
A-3: The API address is set with `NEXT_PUBLIC_API_BASE`, which includes `/api/v1` (default `http://localhost:8000/api/v1`) — this matches `web/.env.example`, already on main.
A-4: Steering strength runs from 0 to 3 in steps of 0.25, default 1 (the guide says 0 to 10, default 4) — the API contract caps `strength` at 3 (`SteerRequest`, `le=3`), and the contract outranks the guide.
A-5: `/inspect` requests time out after 30 s (reads 10 s, `/steer` 60 s, as in the guide) — the guide does not list inspect, and it runs the model, so it gets more than a read.
A-6: Percent values below 0.1 keep three decimals, otherwise one — the real `dead_pct` is 0.0081, which one decimal would show as a misleading 0.0%.
A-7: The "sample data" marks come from `is_synthetic`, and the "simulated" marks on Inspect and Steer come from `mode` — so real data served in mock mode is never labelled as fake.
A-8: Tokens are shown exactly as stored, never trimmed, with `↵`, `⇥`, `␠` and `∅` for invisible characters; leading spaces stay spaces — section 10.3 and gotcha 5.
A-9: Two extra screens: a landing page at `/` and a Microscope at `/microscope` that shows every feature as a dot placed by density rank (most frequent in the centre) — the positions carry no claim of similarity; they will switch to the 2D projection when W7 delivers it.
A-10: Example tokens carry a `title` and `aria-label` with their activation, but are not tab stops — a detail page holds about 2,000 tokens, which would make keyboard navigation unusable.
A-11: Headings and helper text not covered by the copy deck are kept short; every copy-deck key in use is taken word for word from `web/src/lib/copy.ts`.
A-12: The Microscope asks `/inspect` for 20 features per token (the API maximum) so the field shows a fuller picture of each token; the Inspect page defaults to 5.

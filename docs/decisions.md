# Decisions

Format: `A-<n>: <decision> — <reason>` (docs/APP_BUILD_GUIDE.md section 3.2).

## Web (Anand)

A-1: The web app uses Next.js 16 and React 19 with CSS modules and hand-drawn SVG and canvas charts, not Next 14, Tailwind, Zustand or Recharts — it was built and merged (PR #3) before the guide was published, it builds and passes the core path, and switching now adds risk with no visible gain.
A-2: Black-and-white theme with the Geist font instead of the guide's lime accent and system fonts — chosen by the web owner; `next/font` downloads Geist once at build time and serves it from the app, so the running app needs no network.
A-3: The API address is set with `NEXT_PUBLIC_API_BASE`, which includes `/api/v1` (default `http://localhost:8000/api/v1`) — this matches `web/.env.example`, already on main.
A-4: Steering strength runs from 0 to 3 in steps of 0.25, default 1 (the guide says 0 to 10, default 4) — the API contract caps `strength` at 3 (`SteerRequest`, `le=3`), and the contract outranks the guide.
A-5: `/inspect` requests time out after 30 s (reads 10 s, `/steer` 60 s, as in the guide) — the guide does not list inspect, and it runs the model, so it gets more than a read.
A-6: Percent values below 1 keep two significant digits, otherwise one decimal — the real `dead_pct` is 0.0081 (1 of 12,288 features), which one decimal would show as a misleading 0.0% and three decimals as 0.008%.
A-7: The "sample data" marks come from `is_synthetic`, and the "simulated" marks on Inspect and Steer come from `mode` — so real data served in mock mode is never labelled as fake.
A-8: Tokens are shown exactly as stored, never trimmed, with `↵`, `⇥` and `␠` for invisible characters; leading spaces stay spaces — section 10.3 and gotcha 5. An empty token shows as a small, clickable `·` chip instead of `∅` — after the byte-stream decoding fix (PR #4) the leading byte-pieces of characters such as ’ arrive as empty strings, and `·` reads as a quiet joiner rather than an error.
A-9: Two extra screens: a landing page at `/` and a Microscope at `/microscope` that shows every feature as a dot placed by density rank (most frequent in the centre) — the positions carry no claim of similarity; they will switch to the 2D projection when W7 delivers it.
A-10: Example tokens carry a `title` and `aria-label` with their activation, but are not tab stops — a detail page holds about 2,000 tokens, which would make keyboard navigation unusable.
A-11: Headings and helper text not covered by the copy deck are kept short; every copy-deck key in use is taken word for word from `web/src/lib/copy.ts`.
A-12: The Microscope asks `/inspect` for 20 features per token (the API maximum) so the field shows a fuller picture of each token; the Inspect page defaults to 5.
A-13: The header's status pill shows only the server state (Mock, Live, Sample data, Offline); the model chip (`gpt2 · L8 · 12,288 · k=32`) unfolds inside it on hover or keyboard focus and is always in its `aria-label`, and the header has no "Open microscope" button (Microscope is the first link) — section 10.3 lists the chip on its own; folding it keeps the bar short while the same facts stay one hover away.
A-14: The landing page opens with a self-playing tour of the real data (all features, one `/inspect` sentence, three named features, one saved steering run, `ce_recovered`) that loops and can be paused or jumped by chapter — the guide has no landing screen (see A-9); it starts paused for people who ask for reduced motion.
A-15: On the landing page the header and its banner slide out of the way while the tour fills the screen, and come back as soon as you scroll, move the mouse to the top edge, or tab into them; the tour shows the same banner text (mock, sample data or offline) at its top edge in the meantime, so simulated results are never on screen without it (gotcha 14).

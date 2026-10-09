NeuronLens: Build Phases
Oct 8, 2026 · @Prabodh
NeuronLens is built in five phases; a phase ends only when its exit gate passes, and
nobody starts the next phase's work on main before then. This doc goes with the
Developer Handbook, which holds the contract, templates and role guides.
Phases at a glance
Stretch items (S1 to S5) happen only inside phase 4 or later if a person is ahead; they
never delay a gate.
Phase details
Phase 1: Scaffold and contract
Prabodh
Phase Goal Main people Exit gate

1. Scaffold and
   contract
   repo, four branches,
   schema, fixtures,
   templates, real DB shared
   Prabodh Mirdul and Anand can
   clone, run mock mode
   and open the real DB
2. Core in
   mock mode
   server endpoints and
   screens C1 to C5 on
   fixture data
   Mirdul, Anand the whole core path
   works in mock mode
   from a clean clone
3. Real mode
   and names
   real Runtime and DB
   wired in, final evals, auto
   naming (C6)
   all three core path works on
   Prabodh's GPU laptop;
   metrics page shows DB
   numbers; most features
   named
4. Standout
   and Wow
   X1 to X7, then W4, W5,
   W7, W8
   Mirdul, Anand;
   Prabodh for data
   each feature has a real
   demo example and an
   honest limit on screen
5. Freeze and
   demo
   rehearsal, backup
   recording, claims check,
   final merge
   Prabodh leads demo runs from main
   end to end, with the
   fallback tested
   NeuronLens: Build Phases
   Page 1 of 4
   Mirdul and Anand
   Exit gate: both confirm in the group chat that they cloned and opened the DB.
   Phase 2: Core in mock mode
   Mirdul (branch mirdul)
   Anand (branch anand)
   Prabodh (branch prabodh)
   Exit gate: a clean clone runs server and web in mock mode and the core path (browse,
   detail, inspect, steer, metrics) works. Prabodh merges all three branches into main.
   Phase 3: Real mode and names
   Mirdul
   Anand
   Push the ML code, contracts/ (schema, fixtures/sample.db), docs, .github/
   templates, .env.example, .gitignore, .gitattributes.
   Create the four branches: main, prabodh, mirdul, anand.
   Add Mirdul and Anand as GitHub collaborators.
   Send neuronlens.db and the checkpoint by drive link, with the WhatsApp
   instructions.
   Clone, check out your branch, read the Developer Handbook sections 3, 5 and your role
   guide.
   Open the real DB read-only and run one query (for example, feature 1863).
   FastAPI app: /health, /features, /features/{id}, /features/random, /inspect,
   /steer, /steering-examples, /runs, /metrics against fixtures.
   Error format, CORS, server tests; commit contracts/openapi.json.
   Scaffold Next.js with generated types from openapi.json.
   Explorer (C1), Feature Detail (C2), Metrics (C5), Text Inspector (C3), Steering Panel
   (C4), each with loading, empty and error states.
   Review the contract against what the server returns; answer questions the same day.
   Write run_evals.py and start the FVU comparison against the published GPT-2 SAE.
   Real mode: NEURONLENS_MODE=real loads the Runtime and the real DB.
   Write and run auto_label.py (never overwrites human labels).
   NeuronLens: Build Phases
   Page 2 of 4
   NeuronLens: Build Phases
   Point the app at the real DB; remove any text that assumed fixture data; show the
   simulated tag only in mock mode.
   Prabodh
   Run the 256-sequence eval and update the DB metrics.
   Spot-check 50 auto-named features; mark wrong ones
   Verify real-mode inspect and steer on the GPU laptop.
   ambiguous .
   Exit gate: the core path works in real mode on Prabodh's laptop; the Metrics page
   matches the DB; most features have names.
   Phase 4: Standout and Wow
   Order: X1 steering check, X3 seed-matched compare, X2 strength sweep with
   degeneration warning, X4 prompt-only baseline, X5 Limits box, then X6 sentence report
   and X7 compare, then W4, W5, W7, W8.
   Mirdul: the endpoints from Handbook section 5.4, with tests and a documented
   degeneration rule.
   Anand: UI for each feature as soon as its endpoint exists; Limits box shows the honest
   limits from Handbook section 8.
   Prabodh: co-fire counts (W4) and 2D projection (W7) as data files; stretch training (S4,
   S2) if the GPU is free.
   Exit gate: each shipped feature has a real demo example in the DB or in
   demo/ , and an
   honest limit on screen. Anything that fails the gate is hidden or removed.
   Phase 5: Freeze and demo
   No new features. Bug fixes, copy fixes and rehearsal only.
   Anand finalises the demo script and records a backup screen recording in
   Claims check on every screen against the Handbook claims policy.
   demo/ .
   Full rehearsal on Prabodh's laptop, including the fallback to curated steering examples.
   Prabodh makes the final merge to
   main ; the demo runs from
   main only.
   Exit gate: two full rehearsals pass back to back with no manual fixes.
   Rules between phases
   Merges: only Prabodh merges into
   main , and only after a phase's exit gate passes and
   the Handbook merge checklist is ticked.
   Page 3 of 4
   NeuronLens: Build Phases
   Pull main first: after every merge, Mirdul and Anand pull
   main into their branch before
   continuing.
   Contract changes: announced in the group chat before any code; only Prabodh edits
   contracts/schema.sql and
   contracts/openapi.json .
   Waiting on someone: keep building against mock mode and fixtures. A blocked person
   never waits idle for a merge.
   Stuck for more than a few hours: post in the group chat with the error text, the branch
   and what you tried.
   Behind schedule: cut scope from the bottom of the list (Stretch, then Wow); never cut
   the Limits box or the steering check, since they are what makes us different.
   No fake content: no dummy data, no unsupported claim, no half-finished feature on
   screen at any phase.
   Page 4 of 4

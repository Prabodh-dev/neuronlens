# Limits box
User sees: a "What it does not do." box with eight numbered limits, on the landing page (above "Look closer.") and at the bottom of the Metrics page.
Endpoint / data used: `/health` (model, layer, feature count), `/metrics` (`ce_recovered`, `fvu`), `/features?status=named&limit=1` (`total`); copy-deck keys `note.limit` and `note.steering`; four limits seen in the real steering-check runs (docs/features/steering-check.md and strength-sweep.md): no room at the top for a feature near its peak, switched off still reads about the same while the concept word is in the text, the loop warning catches repetition not every drift, results depend on prompt and strength with one feature and one seed.
Recorded example (feature ids, prompt, strength): none needed; the numbers come from `/metrics` on the real DB, never typed in.
Honest limit shown on screen: the box is the limit. A line whose number the server does not return is left out, never filled with a guess.
Status: removed from the UI (web owner's call); each check on the Steer page still shows its own limit under its result

# X5: Limits box
Owner: Anand
Branch: anand
User sees: a "What it does not do." box with five numbered limits, on the landing page (above "Look closer.") and at the bottom of the Metrics page.
Endpoint / data used: `/health` (model, layer, feature count), `/metrics` (`ce_recovered`, `fvu`), `/features?status=named&limit=1` (`total`); copy-deck keys `note.limit` and `note.steering`.
Real demo example (feature ids, prompt, strength): none needed; on the real DB it reads 98.6% of next-word ability kept, 16.3% of variance unexplained, 21 of 12,288 features named, layer 8 of gpt2.
Honest limit shown on screen: the box is the limit. A line whose number the server does not return is left out, never filled with a guess. The wording is checked against Handbook section 8 once the Handbook is in the repo.
Status: working

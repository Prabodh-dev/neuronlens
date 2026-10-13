# Seed-matched compare
User sees: The shared start of the baseline and steered text, the token where they begin to differ, and the text from there in each run.
Endpoint / data used: /api/v1/steer/compare
Recorded example (feature ids, prompt, strength): feature 13, prompt "The acrobat hung", amplify 0.5, seed 42; divergence at new token 4.
Honest limit shown on screen: The raw percent of changed tokens overstates the direct effect, because everything after the divergence drifts. The screen shows the position of the divergence and the tokens changed from there. One feature, one prompt, one seed.
Status: working (ML and server); UI working: the Compare tab on the Steer page (shared start once, then both runs from the split; the raw percent is not shown)

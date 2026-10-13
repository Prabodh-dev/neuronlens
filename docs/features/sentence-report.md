# Sentence report
User sees: Inspect page, "By sentence" tab: each sentence of the text with its most active features, a strength bar for each (relative_to_max: share of the feature's peak in the dictionary) and how many tokens it was active on.
Endpoint / data used: `POST /api/v1/inspect/report` (`sentences[].features[]`: `max_activation`, `relative_to_max`, `tokens_active`).
Recorded example (feature ids, prompt, strength): Real-mode check: #1863 ranks first on the Supreme Court sentence.
Honest limit shown on screen: "Ranked by raw activation, so word-piece and common-word features often come first. Sentences split at . ! ? and new lines; abbreviations such as 'U.S.' can split early. It shows which features were active, not what the sentence means."
Status: working (server and UI), including the optional max_density filter ("Hide very common features") and sort ("Rank by": Strongest or Closest to its own peak)

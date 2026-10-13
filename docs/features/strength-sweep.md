# Strength sweep
User sees: The same prompt steered at several strengths, one output per strength, each with a warning when the text loops or breaks up.
Endpoint / data used: /api/v1/steer/sweep
Recorded example (feature ids, prompt, strength): feature 3446, prompt "The band walked on stage and", seed 42: 0.5 stays coherent, 1.0 drifts to "video" and the activation reads 0.00. Also feature 13, prompt "The acrobat hung": loop warning at 1.0.
Honest limit shown on screen: The loop warning catches repetition, not all poor text, and a result marked ok can still be poor. Results depend on the prompt and the strength (feature 3446 went from 0.00 to 12.44 at 0.5 and read 0.00 at 1.0). This is one feature and one seed, not a general rule.
Status: working (ML and server); UI working: the Every strength tab on the Steer page, seed shown, a "Breaks down" tag with the measured reason

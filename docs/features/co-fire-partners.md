# Co-fire partners
User sees: For one feature, the features that fired on the same tokens most often in a sample, with the counts and the sample size.
Endpoint / data used: `GET /api/v1/features/{feature_id}/cofire`, reading the co-fire data file (data/cofire.json, built by ml/scripts/build_cofire.py).
Recorded example (feature ids, prompt, strength): feature 1863 ("supreme – supreme court") with partners "supreme – court or command" (158 co-fires, Jaccard 0.17) and "courts – legal system" (90); 3,000,000-token sample, seed 999.
Honest limit shown on screen: The features fired on the same tokens in a sample; this does not show that they share a meaning. The evidence is thinner for rare features, and small samples are weak (a 500,000-token sample gave counts of only 5 to 20 for the same features). The screen shows the sample size and the counts.
Status: data file built; endpoint working; UI working: a "Fired on the same tokens" panel on the feature page, sample size and counts from the API, a partner linked only when in_dictionary is true, the panel hidden when available is false

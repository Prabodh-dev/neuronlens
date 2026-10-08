PRAGMA foreign_keys = ON;

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- required keys: schema_version, run_id, model_name, hook_layer, d_model, d_sae, k, created_at, is_synthetic ("true"|"false")

CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  model_name TEXT NOT NULL,
  hook_layer INTEGER NOT NULL,
  d_model INTEGER NOT NULL,
  d_sae INTEGER NOT NULL,
  k INTEGER NOT NULL,
  n_train_activations INTEGER,
  checkpoint_name TEXT,
  created_at TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE features (
  feature_id INTEGER PRIMARY KEY,
  label TEXT,
  label_status TEXT NOT NULL DEFAULT 'unlabelled' CHECK (label_status IN ('unlabelled','named','ambiguous')),
  density REAL NOT NULL,
  max_activation REAL NOT NULL,
  mean_activation REAL,
  is_dead INTEGER NOT NULL DEFAULT 0 CHECK (is_dead IN (0,1)),
  notes TEXT
);
CREATE INDEX idx_features_label ON features(label);
CREATE INDEX idx_features_density ON features(density);
CREATE INDEX idx_features_maxact ON features(max_activation);

CREATE TABLE feature_examples (
  example_id INTEGER PRIMARY KEY AUTOINCREMENT,
  feature_id INTEGER NOT NULL REFERENCES features(feature_id),
  rank INTEGER NOT NULL,
  tokens_json TEXT NOT NULL,
  activations_json TEXT NOT NULL,
  peak_index INTEGER NOT NULL,
  max_activation REAL NOT NULL,
  UNIQUE (feature_id, rank)
);
CREATE INDEX idx_examples_feature ON feature_examples(feature_id);

CREATE TABLE feature_logits (
  feature_id INTEGER NOT NULL REFERENCES features(feature_id),
  direction TEXT NOT NULL CHECK (direction IN ('promoted','suppressed')),
  rank INTEGER NOT NULL,
  token TEXT NOT NULL,
  score REAL NOT NULL,
  PRIMARY KEY (feature_id, direction, rank)
);

CREATE TABLE metrics (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  name TEXT NOT NULL,
  value REAL NOT NULL,
  PRIMARY KEY (run_id, name)
);
-- names: fvu, l0, dead_pct, ce_original, ce_zero_ablation, ce_reconstructed, ce_recovered

CREATE TABLE density_hist (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  bin_lo REAL NOT NULL,
  bin_hi REAL NOT NULL,
  count INTEGER NOT NULL,
  PRIMARY KEY (run_id, bin_lo)
);

CREATE TABLE training_curve (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  step INTEGER NOT NULL,
  loss REAL NOT NULL,
  fvu REAL,
  dead_pct REAL,
  PRIMARY KEY (run_id, step)
);

CREATE TABLE steering_examples (
  example_id INTEGER PRIMARY KEY AUTOINCREMENT,
  feature_id INTEGER NOT NULL REFERENCES features(feature_id),
  mode TEXT NOT NULL CHECK (mode IN ('amplify','ablate')),
  strength REAL NOT NULL,
  prompt TEXT NOT NULL,
  seed INTEGER NOT NULL,
  baseline_text TEXT NOT NULL,
  steered_text TEXT NOT NULL,
  is_curated INTEGER NOT NULL DEFAULT 0 CHECK (is_curated IN (0,1))
);
CREATE INDEX idx_steer_feature ON steering_examples(feature_id);

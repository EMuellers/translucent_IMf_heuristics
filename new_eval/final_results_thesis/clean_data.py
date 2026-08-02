import re
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

BASE_RESULTS_PATH = Path(r"C:\Users\elias\Desktop\final_results_thesis")
PROCESSED_OUTPUT_PATH = Path(
    r"C:\Users\elias\Desktop\final_results_thesis\processed_results"
)

LOG_NAMES = ["Sepsis", "road_traffic_fine", "hospital_billing"]
NOISE_TYPES = [
    "base",
    "remove_enabled",
    "remove_events",
    "add_enabled",
    "add_events_no_trans",
]

# Normal evaluation metrics (from single-run evaluations)
NORMAL_METRIC_BASES = [
    "fitness",
    "translucent_fitness",
    "precision",
    "translucent_precision",
    "f_1_score",
    "translucent_f_1_score",
    "simplicity",
    "fallthrough_count",  # Excluded automatically for Petrify
]

# 10-fold generalization metric mappings
TENFOLD_RENAME_MAP = {
    "fitness": "generalization",
    "translucent_fitness": "translucent_generalization",
}

# Shorthand mapping for arc heuristic names
SHORTEN_MAP = {
    "confidence": "conf.",
    "dependency score": "dep.",
    "support": "sup.",
    "parallel relationship frequency": "par.",
    "exclusive choice frequency": "excl.",
}


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------


def apply_shorthand(val: str) -> str:
  """Converts long heuristic parameter names into their defined shorthands."""
  if not isinstance(val, str) or val in ("False", "True"):
    return val

  clean_val = val.replace("_", " ")
  for long_name, short_name in SHORTEN_MAP.items():
    if long_name in clean_val:
      clean_val = clean_val.replace(long_name, short_name)

  return clean_val


def calc_gen_score(series: pd.Series) -> float:
  """Calculates generalization score across 10 folds.

  If any fold has a value of -1.0 (failed run), returns -1.0. Otherwise
  returns the mean across the 10 folds.
  """
  if (series == -1.0).any():
    return -1.0
  return float(series.mean())


def extract_parameters(file_stem: str) -> tuple[str, dict]:
  """Extracts human-readable configuration label and parameter key-value pairs."""
  clean_stem = (
      file_stem.replace("IMf_10fold_translucent_", "")
      .replace("IMf_results_translucent_", "")
      .replace("IMf_10fold_", "")
      .replace("IMf_results_", "")
  )

  keywords = [
      "translucent_self_loops",
      "strict_end_activities",
      "parallel_end_activities_heuristic",
      "remove_arcs_heuristics",
      "add_arcs_heuristics",
  ]
  pattern = r"(?:_)(" + "|".join(re.escape(k) for k in keywords) + r")"
  parts = re.split(pattern, clean_stem)

  param_dict = {}
  if len(parts) > 1:
    # Minor/delta boolean & heuristic parameters structure
    first_key, first_value = parts[0].rsplit("_", 1)
    param_dict[first_key] = first_value.lstrip("_")
    for i in range(1, len(parts), 2):
      k = parts[i]
      v = parts[i + 1].lstrip("_")
      param_dict[k] = v

    # Apply shorthand mapping to heuristic columns
    for h_key in [
        "remove_arcs_heuristics",
        "add_arcs_heuristics",
        "add_arc_heuristics",
    ]:
      if h_key in param_dict:
        param_dict[h_key] = apply_shorthand(param_dict[h_key])

    active_params = [f"{k}={v}" for k, v in param_dict.items() if v != "False"]
    config_label = (
        " | ".join(active_params) if active_params else "No Heuristics"
    )
  else:
    # Custom parameter structure
    config_label = clean_stem.replace("_", " ")
    param_dict["config_raw"] = clean_stem

  return config_label, param_dict


def extract_normal_metrics(df_norm: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
  """Extracts all normal evaluation metric columns from a single-run DataFrame."""
  select_cols = ["threshold"] if "threshold" in df_norm.columns else []
  rename_dict = {}

  for base in NORMAL_METRIC_BASES:
    col = f"{base}_{suffix}" if suffix else base
    if col in df_norm.columns:
      select_cols.append(col)
      rename_dict[col] = base

  return df_norm[select_cols].rename(columns=rename_dict)


def aggregate_10fold_metrics(df_10f: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
  """Aggregates 10-fold metric columns strictly for generalization and translucent_generalization."""
  agg_dict = {}
  rename_dict = {}

  for base, target_col in TENFOLD_RENAME_MAP.items():
    col = f"{base}_{suffix}" if suffix else base
    if col in df_10f.columns:
      agg_dict[col] = calc_gen_score
      rename_dict[col] = target_col

  if "threshold" in df_10f.columns:
    df_agg = df_10f.groupby("threshold", as_index=False).agg(agg_dict)
    return df_agg.rename(columns=rename_dict)
  else:
    row = {
        target_col: calc_gen_score(df_10f[col])
        for col, target_col in [
            (f"{b}_{suffix}" if suffix else b, t_col)
            for b, t_col in TENFOLD_RENAME_MAP.items()
        ]
        if col in df_10f.columns
    }
    return pd.DataFrame([row])


def apply_shorthands_to_df(df: pd.DataFrame) -> pd.DataFrame:
  """Ensures remove_arcs_heuristics and add_arcs_heuristics columns have shorthand names."""
  target_cols = [
      col
      for col in df.columns
      if col
      in (
          "remove_arcs_heuristics",
          "add_arcs_heuristics",
          "add_arc_heuristics",
      )
  ]
  for col in target_cols:
    df[col] = df[col].astype(str).apply(apply_shorthand)
  return df


# ------------------------------------------------------------------
# Main Processing & Aggregation Routine
# ------------------------------------------------------------------


def process_and_save_datasets():
  """Combines normal evaluations and 10-fold generalization scores into:

  processed_results/{dataset}/{noise_type}/[petrify.csv | IMf.csv | IMtf.csv |
  IMts.csv]
  """
  processed_count = 0

  for log in LOG_NAMES:
    for noise in NOISE_TYPES:
      target_dir = PROCESSED_OUTPUT_PATH / log / noise

      # ==============================================================
      # 1. Process Petrify (petrify.csv)
      # ==============================================================
      petrify_norm_path = (
          BASE_RESULTS_PATH
          / log
          / "results"
          / "petrify"
          / noise
          / "petrify_results.csv"
      )
      petrify_10f_path = (
          BASE_RESULTS_PATH
          / log
          / "results"
          / "petrify_10fold"
          / noise
          / "petrify_10fold_results.csv"
      )

      if petrify_norm_path.exists() or petrify_10f_path.exists():
        petrify_data = {}

        if petrify_norm_path.exists():
          df_p_norm = pd.read_csv(petrify_norm_path)
          for base in NORMAL_METRIC_BASES:
            if base in df_p_norm.columns and base != "fallthrough_count":
              petrify_data[base] = df_p_norm[base].iloc[0]

        if petrify_10f_path.exists():
          df_p_10f = pd.read_csv(petrify_10f_path)
          for base, target_col in TENFOLD_RENAME_MAP.items():
            if base in df_p_10f.columns:
              petrify_data[target_col] = calc_gen_score(df_p_10f[base])

        petrify_res = pd.DataFrame([petrify_data])
        target_dir.mkdir(parents=True, exist_ok=True)
        petrify_res.to_csv(target_dir / "petrify.csv", index=False)

      # ==============================================================
      # 2. Process Baseline IMf (IMf.csv)
      # ==============================================================
      imf_norm_path = (
          BASE_RESULTS_PATH
          / log
          / "results"
          / "IMf"
          / noise
          / "IMf_results.csv"
      )
      imf_10f_path = (
          BASE_RESULTS_PATH
          / log
          / "results"
          / "IMf_10fold"
          / noise
          / "IMf_10fold_results.csv"
      )

      if imf_norm_path.exists() and imf_10f_path.exists():
        df_imf_norm = pd.read_csv(imf_norm_path)
        df_imf_10f = pd.read_csv(imf_10f_path)

        norm_sub = extract_normal_metrics(df_imf_norm)
        agg_10f = aggregate_10fold_metrics(df_imf_10f)

        imf_res = pd.merge(norm_sub, agg_10f, on="threshold").sort_values(
            "threshold"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        imf_res.to_csv(target_dir / "IMf.csv", index=False)

      # ==============================================================
      # 3. Process Translucent Configs (IMtf.csv & IMts.csv)
      # ==============================================================
      exp_pairs = [
          (
              BASE_RESULTS_PATH / log / "results" / "translucent" / noise,
              BASE_RESULTS_PATH
              / log
              / "results"
              / "translucent_minor_10fold"
              / noise,
          ),
          (
              BASE_RESULTS_PATH / log / "results" / "translucent_delta_all" / noise,
              BASE_RESULTS_PATH
              / log
              / "results"
              / "translucent_delta_10fold"
              / noise,
          ),
      ]

      imtf_list = []
      imts_list = []

      for norm_folder, tenfold_folder in exp_pairs:
        if not tenfold_folder.exists():
          continue

        for csv_10f_file in tenfold_folder.glob(
            "IMf_10fold_translucent_*.csv"
        ):
          csv_norm_file = norm_folder / csv_10f_file.name.replace(
              "IMf_10fold_", "IMf_results_"
          )

          df_10f_raw = pd.read_csv(csv_10f_file)
          label, params = extract_parameters(csv_10f_file.stem)

          # Aggregate 10-fold generalization metrics
          df_10f_tf = aggregate_10fold_metrics(df_10f_raw, suffix="tf")
          df_10f_ts = aggregate_10fold_metrics(df_10f_raw, suffix="ts")

          # Merge with normal single-run metrics if file exists
          if csv_norm_file.exists():
            df_norm_raw = pd.read_csv(csv_norm_file)
            df_norm_tf = extract_normal_metrics(df_norm_raw, suffix="tf")
            df_norm_ts = extract_normal_metrics(df_norm_raw, suffix="ts")

            merged_tf = pd.merge(
                df_norm_tf, df_10f_tf, on="threshold"
            ).sort_values("threshold")
            merged_ts = pd.merge(
                df_norm_ts, df_10f_ts, on="threshold"
            ).sort_values("threshold")
          else:
            merged_tf = df_10f_tf.copy()
            merged_ts = df_10f_ts.copy()

          # Prepend configuration label and append individual parameter columns
          merged_tf.insert(0, "configuration", label)
          merged_ts.insert(0, "configuration", label)

          for p_key, p_val in params.items():
            merged_tf[p_key] = p_val
            merged_ts[p_key] = p_val

          # Apply shorthands to arc heuristic columns
          merged_tf = apply_shorthands_to_df(merged_tf)
          merged_ts = apply_shorthands_to_df(merged_ts)

          imtf_list.append(merged_tf)
          imts_list.append(merged_ts)

      if imtf_list:
        target_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(imtf_list, ignore_index=True).to_csv(
            target_dir / "IMtf.csv", index=False
        )

      if imts_list:
        target_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(imts_list, ignore_index=True).to_csv(
            target_dir / "IMts.csv", index=False
        )

      processed_count += 1

  print(f"Aggregation complete. Output saved to: {PROCESSED_OUTPUT_PATH}")


if __name__ == "__main__":
  process_and_save_datasets()
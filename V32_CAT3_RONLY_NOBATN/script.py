"""제출용 추론 스크립트. V24 원본에서 CatBoost 부분만 3시드 평균으로 교체.

V24 대비 diff:
  - cat.pkl (모델 1개)  →  cats.pkl (모델 N개, predict_proba 평균)
  - 그 외 피처·블렌드·보정 전부 동일

make_submission.py 가 이 파일의 add_features() 를 그대로 import 해서 학습에 쓴다.
학습/추론 피처 정의가 물리적으로 하나이므로 불일치가 발생할 수 없다.
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
EPS = 1e-6

# ── 보정 상수 (V24 계승) ───────────────────────────────
CAT_WEIGHT = 0.6
LOGIT_SCALE = 1.09
LOGIT_DELTA = -0.0521

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def model_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "model", filename),
        os.path.join(script_dir, filename),
        os.path.join("./model", filename),
        f"./{filename}",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Model file not found for {filename}. Tried: {candidates}")


def add_features(df):
    """V24 원본 add_features 그대로 (89피처)."""
    df = df.copy()
    balls = df["balls_before"]
    strikes = df["strikes_before"]
    outs = df["outs_before"]
    score_diff = df["score_diff_pitcher_team"]

    df["count_state"] = balls.astype("string") + "_" + strikes.astype("string")
    df["base_out_state"] = df["base_state"].astype("string") + "_" + outs.astype("string")
    df["platoon"] = df["pitcher_hand"].astype("string") + "_" + df["batter_hand"].astype("string")

    df["is_full_count"] = ((balls == 3) & (strikes == 2)).astype("int8")
    df["is_two_strike"] = (strikes == 2).astype("int8")
    df["is_three_ball"] = (balls == 3).astype("int8")
    df["pitcher_ahead"] = (strikes > balls).astype("int8")
    df["pitcher_behind"] = (balls > strikes).astype("int8")
    df["count_pressure"] = (balls + strikes).astype("int8")

    df["is_late_inning"] = (df["inning"] >= 7).astype("int8")
    df["is_extra_inning"] = (df["inning"] >= 10).astype("int8")
    df["abs_score_diff"] = score_diff.abs()
    df["is_tie_game"] = (score_diff == 0).astype("int8")
    df["is_close_game"] = (score_diff.abs() <= 1).astype("int8")
    df["pitcher_team_leading"] = (score_diff > 0).astype("int8")
    df["pitcher_team_trailing"] = (score_diff < 0).astype("int8")

    df["has_runner"] = (df["num_runners_on"] > 0).astype("int8")
    df["is_scoring_position"] = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype("int8")
    df["runner_pressure"] = df["num_runners_on"] * (outs + 1)
    df["late_close"] = (df["is_late_inning"] & df["is_close_game"]).astype("int8")
    df["winexp_gap_home_away"] = df["home_win_expectancy"] - df["away_win_expectancy"]

    df["same_hand_matchup"] = (df["pitcher_hand"].astype("string") == df["batter_hand"].astype("string")).astype("int8")
    df["teams_same"] = (df["pitcher_team_id"] == df["batter_team_id"]).astype("int8")

    p_succ = df["asof_pitcher_success_rate"]
    p_mid = df["asof_pitcher_middle_rate"]
    p_ball = df["asof_pitcher_ball_rate"]
    p_rev = df["asof_pitcher_reverse_rate"]
    p_prev5 = df["asof_pitcher_prev5_game_success_rate"]

    mix_cols = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
    mix = df[mix_cols].fillna(0).clip(lower=0)
    mix_sum = mix.sum(axis=1)
    norm = mix.div(mix_sum.replace(0, np.nan), axis=0).fillna(0)
    entropy = -(norm * np.log(norm.replace(0, np.nan))).sum(axis=1).fillna(0)
    df["pitchmix_entropy"] = entropy
    df["pitchmix_max_rate"] = norm.max(axis=1)
    df["fastball_minus_breaking"] = df["asof_pitcher_fastball_rate"] - df["asof_pitcher_breaking_rate"]
    df["fastball_minus_offspeed"] = df["asof_pitcher_fastball_rate"] - df["asof_pitcher_offspeed_rate"]
    df["breaking_minus_offspeed"] = df["asof_pitcher_breaking_rate"] - df["asof_pitcher_offspeed_rate"]
    df["is_fastball_heavy"] = (df["asof_pitcher_fastball_rate"] >= 0.6).astype("int8")
    df["is_balanced_mix"] = (df["pitchmix_max_rate"] <= 0.5).astype("int8")

    must_strike = (balls == 3).astype("int8")
    waste = ((strikes == 2) & (balls <= 1)).astype("int8")
    bases_loaded = ((df["runner_on_1b"] == 1) & (df["runner_on_2b"] == 1) & (df["runner_on_3b"] == 1)).astype("int8")
    high_li = (df["li"] >= 1.5).astype("int8")

    df["ix_must_ball"] = must_strike * p_ball
    df["ix_must_middle"] = must_strike * p_mid
    df["ix_must_success"] = must_strike * p_succ
    df["ix_waste_ball"] = waste * p_ball
    df["ix_waste_reverse"] = waste * p_rev
    df["ix_risp_success"] = df["is_scoring_position"] * p_succ
    df["ix_risp_middle"] = df["is_scoring_position"] * p_mid
    df["ix_loaded_ball"] = bases_loaded * p_ball
    df["ix_loaded_middle"] = bases_loaded * p_mid
    df["ix_highli_prev5"] = high_li * p_prev5
    df["ix_lateclose_success"] = df["late_close"] * p_succ
    df["ix_samehand_success"] = df["same_hand_matchup"] * p_succ
    return df


def prepare_cat_x(df, features, cat_features):
    out = df[features].copy()
    for col in cat_features:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__NA__")
    return out


def logit(preds):
    preds = np.clip(preds, EPS, 1.0 - EPS)
    return np.log(preds / (1.0 - preds))


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def apply_logit_scale_delta(preds, scale=LOGIT_SCALE, delta=LOGIT_DELTA):
    return sigmoid(scale * logit(preds) + delta)


def main():
    data_dir, out_dir = "./data", "./output"
    test_path = os.path.join(data_dir, "test.csv")
    sample_path = os.path.join(data_dir, "sample_submission.csv")
    out_path = os.path.join(out_dir, "submission.csv")

    cat_artifact = joblib.load(model_path("cats.pkl"))
    lx_artifact = joblib.load(model_path("lx5.pkl"))
    features = cat_artifact["features"]
    cat_features = cat_artifact["cat_features"]

    # 학습 시 저장해둔 보정값을 우선 사용 (없으면 모듈 상수)
    cat_weight = float(cat_artifact.get("cat_weight", CAT_WEIGHT))
    scale = float(cat_artifact.get("logit_scale", LOGIT_SCALE))
    delta = float(cat_artifact.get("logit_delta", LOGIT_DELTA))

    test = pd.read_csv(test_path, encoding="utf-8-sig")
    sub = pd.read_csv(sample_path, encoding="utf-8-sig")
    test_fe = add_features(test)

    missing = [c for c in features if c not in test_fe.columns]
    if missing:
        raise ValueError(f"Missing feature columns in test.csv: {missing[:10]}")

    cat_x = prepare_cat_x(test_fe, features, cat_features)
    pred_cat = np.mean([m.predict_proba(cat_x)[:, 1] for _, m in cat_artifact["models"]], axis=0)

    lx_x = test_fe[lx_artifact["features"]]
    pred_lx = np.mean([m.predict_proba(lx_x)[:, 1] for _, m in lx_artifact["models"]], axis=0)

    raw_blend = cat_weight * pred_cat + (1.0 - cat_weight) * pred_lx
    preds = apply_logit_scale_delta(raw_blend, scale=scale, delta=delta)

    pred_df = pd.DataFrame({ID_COL: test_fe[ID_COL].values, TARGET_COL: preds})
    sub = sub[[ID_COL]].merge(pred_df, on=ID_COL, how="left")
    if sub[TARGET_COL].isna().any():
        raise ValueError("Some sample_submission row_id values were not found in test.csv")

    sub[TARGET_COL] = sub[TARGET_COL].clip(0.0, 1.0)
    os.makedirs(out_dir, exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path} rows={len(sub)}")


if __name__ == "__main__":
    main()

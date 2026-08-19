"""피처 정의.

  build()      확정 26개 + A/B 후보 (1차 실행용, 결과적으로 폐기)
  build_v24()  V24 원본 89개 ± add/drop  (2차 실행 기준선)

2026-08 스크리닝 결과: 89개 > 26개 (+80.1점, 15σ).
GBDT는 '정보량 중복'이어도 '쪼개기 쉬운' 인코딩에서 이득을 본다.
따라서 기준선은 V24 89개이며, 여기서 측정으로 확인된 것만 빼고 더한다.
"""
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 공용 파생 (build / build_v24 양쪽에서 사용)
# ─────────────────────────────────────────────────────────────
MIXCOLS = ["asof_pitcher_fastball_rate",
           "asof_pitcher_breaking_rate",
           "asof_pitcher_offspeed_rate"]


def _pitchmix_norm(df: pd.DataFrame) -> pd.DataFrame:
    mix = df[MIXCOLS].fillna(0).clip(lower=0)
    return mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def _pitchmix_entropy(df: pd.DataFrame) -> pd.Series:
    norm = _pitchmix_norm(df)
    return -(norm * np.log(norm.replace(0, np.nan))).sum(axis=1).fillna(0)


def _pit_we(df: pd.DataFrame) -> np.ndarray:
    """투수팀 기준 기대승률. top_bottom=='T' 일 때 투수가 홈 (일치율 1.0000 확인)."""
    return np.where(df["top_bottom"].eq("T"),
                    df["home_win_expectancy"],
                    df["away_win_expectancy"]).astype("float32")


def _form_delta(df: pd.DataFrame) -> pd.Series:
    """최근 5경기 폼 − 통산 실력. 교차표에서 양방향 독립 기여 확인."""
    return (df["asof_pitcher_prev5_game_success_rate"]
            - df["asof_pitcher_success_rate"])


# ─────────────────────────────────────────────────────────────
# V33: 공식 타깃정의(DACON) 기반 실패모드 분해 + 상황 상호작용
#   control_success 실패 3조건: ①가운데 부근 ②존에서 크게 벗어남 ③포수요구 반대
#   asof_pitcher_middle_rate=①전용, asof_pitcher_reverse_rate=③전용 프록시(공식 컬럼정의 확인).
#   ②는 전용 컬럼이 없어 잔차로 근사(residual_fail_rate). 상세: project memory
#   project_target_definition.md / project_v33_step_plan.md 참조.
# ─────────────────────────────────────────────────────────────
def _residual_fail_rate(df: pd.DataFrame) -> pd.Series:
    """실패조건②(스트존에서 크게 벗어남)의 근사 프록시.
    세 실패모드가 대체로 배타적이라는 가정하에 success/middle/reverse의 잔차로 유도.
    타깃정의에서 직접 유도된 피처라 추측성 피처보다 근거가 강함."""
    return (1 - df["asof_pitcher_success_rate"]
            - df["asof_pitcher_middle_rate"]
            - df["asof_pitcher_reverse_rate"]).astype("float32")


def _residual_fail_rate_prev5(df: pd.DataFrame) -> pd.Series:
    """위와 동일한 근사이나 최근 5경기 창. prev5_reverse_rate 컬럼이 없어
    reverse+gross-miss 실패가 섞인 근사치임에 유의(순수 ②전용 아님)."""
    return (1 - df["asof_pitcher_prev5_game_success_rate"]
            - df["asof_pitcher_prev5_game_middle_rate"]).astype("float32")


def _flag_two_strike(df: pd.DataFrame) -> pd.Series:
    return df["strikes_before"].eq(2)


def _flag_scoring_position(df: pd.DataFrame) -> pd.Series:
    return (df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)


def _flag_late_close(df: pd.DataFrame) -> pd.Series:
    is_late = df["inning"] >= 7
    is_close = df["score_diff_pitcher_team"].abs() <= 1
    return is_late & is_close


def _flag_high_li(df: pd.DataFrame) -> pd.Series:
    return df["li"] >= 1.5


def _flag_same_hand(df: pd.DataFrame) -> pd.Series:
    return df["pitcher_hand"].astype(str) == df["batter_hand"].astype(str)


# 우선순위1: middle/reverse/residual × 기존 ix_*가 아직 안 곱해본 상황
#   (기존 ix_*는 must/waste/risp/loaded/highli_prev5/lateclose_success/samehand_success 뿐,
#    middle·reverse·residual과 2스트라이크/highli/lateclose/count_pressure 조합은 전무했음)
def _ix_2strike_middle(df):
    return (_flag_two_strike(df) * df["asof_pitcher_middle_rate"]).astype("float32")


def _ix_2strike_reverse(df):
    return (_flag_two_strike(df) * df["asof_pitcher_reverse_rate"]).astype("float32")


def _ix_risp_reverse(df):
    return (_flag_scoring_position(df) * df["asof_pitcher_reverse_rate"]).astype("float32")


def _ix_highli_middle(df):
    return (_flag_high_li(df) * df["asof_pitcher_middle_rate"]).astype("float32")


def _ix_highli_reverse(df):
    return (_flag_high_li(df) * df["asof_pitcher_reverse_rate"]).astype("float32")


def _ix_lateclose_middle(df):
    return (_flag_late_close(df) * df["asof_pitcher_middle_rate"]).astype("float32")


def _ix_lateclose_reverse(df):
    return (_flag_late_close(df) * df["asof_pitcher_reverse_rate"]).astype("float32")


def _ix_countpressure_residual(df):
    pressure = (df["balls_before"] + df["strikes_before"]).astype("float32")
    return (pressure * _residual_fail_rate(df)).astype("float32")


# 우선순위2: 플래툰(같은손) 정교화 — 문헌상 실재하는 작은 신호(τ≈0.015, 클러치와 달리 0 아님)
def _ix_samehand_middle(df):
    return (_flag_same_hand(df) * df["asof_pitcher_middle_rate"]).astype("float32")


def _ix_samehand_reverse(df):
    return (_flag_same_hand(df) * df["asof_pitcher_reverse_rate"]).astype("float32")


# 우선순위3: RISP/high-LI × residual — 문헌상(클러치) 신호 기대치는 낮지만 스크리닝 가치는 있음
def _ix_risp_residual(df):
    return (_flag_scoring_position(df) * _residual_fail_rate(df)).astype("float32")


def _ix_highli_residual(df):
    return (_flag_high_li(df) * _residual_fail_rate(df)).astype("float32")


# ─────────────────────────────────────────────────────────────
# V33 확장 (2026-08-17) — 기존 89피처 전수대조 후 남은 공백만 채운 후보들.
# 설계 원칙: GBDT에서 단일 피처의 단조변환(log/sqrt/버킷)은 정보량 0이므로 제외.
#            새로운 조합(차·비·곱·상호작용)만 후보로 둔다.
# 실측 확인(train 147만행):
#   fastball+breaking+offspeed = 1.000  → 믹스 단독 재조합은 완전중복, 상호작용만 유효
#   ball_rate + strike_rate    = 0.814  → 잔여 0.186이 미사용 정보 (inplay_rate)
#   strike_rate vs success 상관 = 0.088 → 89피처 중 파생 0회인 유일한 asof 컬럼
# ─────────────────────────────────────────────────────────────
_EPS = 1e-6

# fit셋에서만 계산해 valid/test에 상수로 씌우는 리그 평균 (log5/EB 계열이 사용).
# run_screen 계열이 build 전에 set_league_mean() 을 호출한다.
LEAGUE_MEAN = None
PITCHER_EB_K = 800.0


def set_league_mean(value: float) -> None:
    """fit 데이터의 타깃 평균을 등록. 검증셋을 보고 잡으면 규칙 5절 위반 소지가 있어
    반드시 fit에서만 계산한 값을 넣는다."""
    global LEAGUE_MEAN
    LEAGUE_MEAN = float(value)


def _require_league_mean() -> float:
    if LEAGUE_MEAN is None:
        raise RuntimeError(
            "LEAGUE_MEAN 미설정: log5/EB 계열 피처를 쓰려면 fit 데이터로 "
            "features.set_league_mean(fit_df[TARGET].mean()) 을 먼저 호출하세요."
        )
    return LEAGUE_MEAN


def _p(df, col):
    return df[col].astype("float32")


def _fail(df):
    """투수 실패율 = 1 - 성공률."""
    return (1.0 - _p(df, "asof_pitcher_success_rate")).clip(lower=_EPS)


# ── G1. 실패모드 분해 (공식 타깃정의 유도) ──────────────────
def _middle_share_of_fail(df):
    return (_p(df, "asof_pitcher_middle_rate") / _fail(df)).astype("float32")


def _reverse_share_of_fail(df):
    return (_p(df, "asof_pitcher_reverse_rate") / _fail(df)).astype("float32")


def _residual_share_of_fail(df):
    return (_residual_fail_rate(df) / _fail(df)).astype("float32")


def _fail_mode_entropy(df):
    """실패가 한 유형(가운데/반대/크게벗어남)에 몰리는지 고르게 퍼지는지."""
    f = _fail(df)
    shares = [
        (_p(df, "asof_pitcher_middle_rate") / f).clip(lower=_EPS, upper=1.0),
        (_p(df, "asof_pitcher_reverse_rate") / f).clip(lower=_EPS, upper=1.0),
        (_residual_fail_rate(df) / f).clip(lower=_EPS, upper=1.0),
    ]
    ent = sum(-(s * np.log(s)) for s in shares)
    return ent.astype("float32")


def _batter_residual_fail(df):
    """타자쪽 잔차 실패(타자는 reverse 컬럼이 없어 middle만 차감)."""
    return (1.0 - _p(df, "asof_batter_success_rate")
            - _p(df, "asof_batter_middle_rate")).astype("float32")


# ── G2. ball/strike 미탐색 축 ───────────────────────────────
def _inplay_rate(df):
    """1 - ball - strike. 실측 평균 0.186의 미사용 잔여 정보."""
    return (1.0 - _p(df, "asof_pitcher_ball_rate")
            - _p(df, "asof_pitcher_strike_rate")).astype("float32")


def _ball_minus_strike(df):
    return (_p(df, "asof_pitcher_ball_rate") - _p(df, "asof_pitcher_strike_rate")).astype("float32")


def _ball_strike_ratio(df):
    return (_p(df, "asof_pitcher_ball_rate")
            / _p(df, "asof_pitcher_strike_rate").clip(lower=_EPS)).astype("float32")


def _strike_x_middle(df):
    """존에 넣되 가운데로 몰리는 성향 = 위험형 스트라이크."""
    return (_p(df, "asof_pitcher_strike_rate") * _p(df, "asof_pitcher_middle_rate")).astype("float32")


def _ball_over_fail(df):
    return (_p(df, "asof_pitcher_ball_rate") / _fail(df)).astype("float32")


# ── G3. 상호작용 공백 (2스트라이크·strike_rate·타자rate 축이 전무했음) ──
def _ix_2strike_success(df):
    return (_flag_two_strike(df) * _p(df, "asof_pitcher_success_rate")).astype("float32")


def _ix_2strike_strike(df):
    return (_flag_two_strike(df) * _p(df, "asof_pitcher_strike_rate")).astype("float32")


def _ix_2strike_residual(df):
    return (_flag_two_strike(df) * _residual_fail_rate(df)).astype("float32")


def _ix_must_reverse(df):
    return (df["balls_before"].eq(3) * _p(df, "asof_pitcher_reverse_rate")).astype("float32")


def _ix_must_strike(df):
    return (df["balls_before"].eq(3) * _p(df, "asof_pitcher_strike_rate")).astype("float32")


def _ix_countpressure_success(df):
    pressure = (df["balls_before"] + df["strikes_before"]).astype("float32")
    return (pressure * _p(df, "asof_pitcher_success_rate")).astype("float32")


def _ix_batter_2strike(df):
    return (_flag_two_strike(df) * _p(df, "asof_batter_success_rate")).astype("float32")


def _ix_batter_samehand(df):
    return (_flag_same_hand(df) * _p(df, "asof_batter_success_rate")).astype("float32")


def _ix_batter_risp(df):
    return (_flag_scoring_position(df) * _p(df, "asof_batter_success_rate")).astype("float32")


def _ix_li_success(df):
    """기존 ix_highli_prev5 는 li>=1.5 이진값만 사용 — 연속 li 곱은 없었음."""
    return (df["li"].astype("float32") * _p(df, "asof_pitcher_success_rate")).astype("float32")


def _ix_outs_success(df):
    return (df["outs_before"].astype("float32") * _p(df, "asof_pitcher_success_rate")).astype("float32")


def _ix_highli_success(df):
    return (_flag_high_li(df) * _p(df, "asof_pitcher_success_rate")).astype("float32")


# ── G4. 투수×타자 매치업 결합 (Bill James log5, 1981) ────────
def _log5_success(df):
    """log5 = (p·b/L) / (p·b/L + (1-p)(1-b)/(1-L)). 투수·타자 rate가 지금까지
    각각 독립 컬럼으로만 들어가 있고 결합항이 전혀 없었음."""
    L = _require_league_mean()
    p = _p(df, "asof_pitcher_success_rate").clip(_EPS, 1 - _EPS)
    b = _p(df, "asof_batter_success_rate").clip(_EPS, 1 - _EPS)
    num = p * b / L
    den = num + (1 - p) * (1 - b) / (1 - L)
    return (num / den.clip(lower=_EPS)).astype("float32")


def _log5_logit(df):
    """log5의 로짓 등가형. 트리 분기에는 이쪽이 더 잘 맞을 수 있어 별도 후보로 둠."""
    L = _require_league_mean()
    p = _p(df, "asof_pitcher_success_rate").clip(_EPS, 1 - _EPS)
    b = _p(df, "asof_batter_success_rate").clip(_EPS, 1 - _EPS)
    lg = lambda v: np.log(v / (1 - v))
    return (lg(p) + lg(b) - np.log(L / (1 - L))).astype("float32")


def _log5_middle(df):
    L_mid = float(np.clip(_require_league_mean() * 0 + 0.14, _EPS, 1 - _EPS))  # middle 리그 기준값
    p = _p(df, "asof_pitcher_middle_rate").clip(_EPS, 1 - _EPS)
    b = _p(df, "asof_batter_middle_rate").clip(_EPS, 1 - _EPS)
    num = p * b / L_mid
    den = num + (1 - p) * (1 - b) / (1 - L_mid)
    return (num / den.clip(lower=_EPS)).astype("float32")


def _pb_success_diff(df):
    return (_p(df, "asof_pitcher_success_rate") - _p(df, "asof_batter_success_rate")).astype("float32")


def _pb_success_prod(df):
    return (_p(df, "asof_pitcher_success_rate") * _p(df, "asof_batter_success_rate")).astype("float32")


def _pb_mid_diff(df):
    return (_p(df, "asof_pitcher_middle_rate") - _p(df, "asof_batter_middle_rate")).astype("float32")


# ── G5. 폼/추세 (prev1·prev3 는 raw로만 존재, 파생 0개였음) ──
def _prev(df, n, kind="success"):
    return _p(df, f"asof_pitcher_prev{n}_game_{kind}_rate")


def _form_trend_1_3(df):
    return (_prev(df, 1) - _prev(df, 3)).astype("float32")


def _form_trend_3_5(df):
    return (_prev(df, 3) - _prev(df, 5)).astype("float32")


def _form_accel(df):
    return ((_prev(df, 1) - _prev(df, 3)) - (_prev(df, 3) - _prev(df, 5))).astype("float32")


def _form_vol(df):
    m = pd.concat([_prev(df, 1), _prev(df, 3), _prev(df, 5)], axis=1)
    return m.std(axis=1).astype("float32")


def _form_range(df):
    m = pd.concat([_prev(df, 1), _prev(df, 3), _prev(df, 5)], axis=1)
    return (m.max(axis=1) - m.min(axis=1)).astype("float32")


def _form_delta_abs(df):
    """form_delta 자체는 이미 시도됨(+1.7, seed_std 13.4) — 부호를 지운 '변동 크기' 버전."""
    return (_prev(df, 5) - _p(df, "asof_pitcher_success_rate")).abs().astype("float32")


def _mid_form_delta(df):
    return (_prev(df, 5, "middle") - _p(df, "asof_pitcher_middle_rate")).astype("float32")


def _mid_form_trend_1_3(df):
    return (_prev(df, 1, "middle") - _prev(df, 3, "middle")).astype("float32")


# ── G6. 시간축 · 표본 비대칭 ────────────────────────────────
def _time_index(df):
    """season(중요도 21.6%)과 game_month가 분리돼 있어 연속 시간축이 없었음."""
    return ((df["season"] - 2019) * 12 + df["game_month"]).astype("float32")


def _batter_pitcher_n_ratio(df):
    """매치업 정보량 비대칭도. raw asof_batter_n 노출은 -14.3으로 손해였으나
    비율 안에 들어가는 것은 raw 노출과 다름 — 그래서 별도 플래그로 분리."""
    return (_p(df, "asof_batter_n") / _p(df, "asof_pitcher_n").clip(lower=_EPS)).astype("float32")


def _mix_coverage(df):
    return (_p(df, "asof_pitcher_pitchmix_n") / _p(df, "asof_pitcher_n").clip(lower=_EPS)).astype("float32")


def _pitcher_succ_eb(df):
    """투수 shrinkage. 타자판(k=5~120)은 전부 실패했으나 투수는 미시도."""
    L = _require_league_mean()
    n = _p(df, "asof_pitcher_n").clip(lower=0.0)
    w = n / (n + PITCHER_EB_K)
    return (w * _p(df, "asof_pitcher_success_rate") + (1 - w) * L).astype("float32")


# ── G7. 상황 인코딩 연속화 ──────────────────────────────────
# RE24 (MLB 2010-2015 표준 런기대값). base_out_state 25범주의 단조 연속 인코딩.
RE24 = {
    ("___", 0): 0.481, ("___", 1): 0.254, ("___", 2): 0.098,
    ("1__", 0): 0.859, ("1__", 1): 0.509, ("1__", 2): 0.224,
    ("_2_", 0): 1.100, ("_2_", 1): 0.664, ("_2_", 2): 0.319,
    ("12_", 0): 1.437, ("12_", 1): 0.884, ("12_", 2): 0.429,
    ("__3", 0): 1.350, ("__3", 1): 0.950, ("__3", 2): 0.353,
    ("1_3", 0): 1.784, ("1_3", 1): 1.130, ("1_3", 2): 0.478,
    ("_23", 0): 1.964, ("_23", 1): 1.376, ("_23", 2): 0.580,
    ("123", 0): 2.292, ("123", 1): 1.541, ("123", 2): 0.752,
}


def _count_signed(df):
    """부호 있는 카운트 차이. pitcher_ahead/behind는 이진, count_pressure는 합만 있었음."""
    return (df["strikes_before"] - df["balls_before"]).astype("float32")


def _runners_weighted(df):
    """num_runners_on 은 단순 개수 — 진루도(득점 근접성) 가중이 없었음."""
    return (df["runner_on_1b"] + 2 * df["runner_on_2b"] + 3 * df["runner_on_3b"]).astype("float32")


def _base_out_re(df):
    keys = list(zip(df["base_state"].astype(str), df["outs_before"].astype(int)))
    return pd.Series([RE24.get(k, np.nan) for k in keys], index=df.index).astype("float32")


def _is_blowout(df):
    return (df["score_diff_pitcher_team"].abs() >= 5).astype("int8")


def _inning_x_li(df):
    return (df["inning"].astype("float32") * df["li"].astype("float32")).astype("float32")


# ── G8. 구종믹스 × 상황 (믹스 합=1이라 단독 재조합은 중복, 상호작용만 유효) ──
def _ix_3ball_fastball(df):
    return (df["balls_before"].eq(3) * _p(df, "asof_pitcher_fastball_rate")).astype("float32")


def _ix_2strike_offspeed(df):
    return (_flag_two_strike(df) * _p(df, "asof_pitcher_offspeed_rate")).astype("float32")


def _ix_2strike_breaking(df):
    return (_flag_two_strike(df) * _p(df, "asof_pitcher_breaking_rate")).astype("float32")


def _mixentropy_x_countpressure(df):
    pressure = (df["balls_before"] + df["strikes_before"]).astype("float32")
    return (_pitchmix_entropy(df).astype("float32") * pressure).astype("float32")


NEW_DERIVED = {
    # 기시도(기각/보류) — 기준선에는 미포함
    "pit_we": _pit_we,
    "form_delta": _form_delta,

    # G1 실패모드 분해
    "residual_fail_rate": _residual_fail_rate,
    "residual_fail_rate_prev5": _residual_fail_rate_prev5,
    "middle_share_of_fail": _middle_share_of_fail,
    "reverse_share_of_fail": _reverse_share_of_fail,
    "residual_share_of_fail": _residual_share_of_fail,
    "fail_mode_entropy": _fail_mode_entropy,
    "batter_residual_fail": _batter_residual_fail,

    # G2 ball/strike 축
    "inplay_rate": _inplay_rate,
    "ball_minus_strike": _ball_minus_strike,
    "ball_strike_ratio": _ball_strike_ratio,
    "strike_x_middle": _strike_x_middle,
    "ball_over_fail": _ball_over_fail,

    # G3 상호작용 공백
    "ix_2strike_success": _ix_2strike_success,
    "ix_2strike_middle": _ix_2strike_middle,
    "ix_2strike_reverse": _ix_2strike_reverse,
    "ix_2strike_strike": _ix_2strike_strike,
    "ix_2strike_residual": _ix_2strike_residual,
    "ix_must_reverse": _ix_must_reverse,
    "ix_must_strike": _ix_must_strike,
    "ix_countpressure_success": _ix_countpressure_success,
    "ix_countpressure_residual": _ix_countpressure_residual,
    "ix_samehand_middle": _ix_samehand_middle,
    "ix_samehand_reverse": _ix_samehand_reverse,
    "ix_batter_2strike": _ix_batter_2strike,
    "ix_batter_samehand": _ix_batter_samehand,
    "ix_batter_risp": _ix_batter_risp,
    "ix_li_success": _ix_li_success,
    "ix_outs_success": _ix_outs_success,
    "ix_risp_reverse": _ix_risp_reverse,
    "ix_risp_residual": _ix_risp_residual,
    "ix_highli_success": _ix_highli_success,
    "ix_highli_middle": _ix_highli_middle,
    "ix_highli_reverse": _ix_highli_reverse,
    "ix_highli_residual": _ix_highli_residual,
    "ix_lateclose_middle": _ix_lateclose_middle,
    "ix_lateclose_reverse": _ix_lateclose_reverse,

    # G4 log5 매치업
    "log5_success": _log5_success,
    "log5_logit": _log5_logit,
    "log5_middle": _log5_middle,
    "pb_success_diff": _pb_success_diff,
    "pb_success_prod": _pb_success_prod,
    "pb_mid_diff": _pb_mid_diff,

    # G5 폼/추세
    "form_trend_1_3": _form_trend_1_3,
    "form_trend_3_5": _form_trend_3_5,
    "form_accel": _form_accel,
    "form_vol": _form_vol,
    "form_range": _form_range,
    "form_delta_abs": _form_delta_abs,
    "mid_form_delta": _mid_form_delta,
    "mid_form_trend_1_3": _mid_form_trend_1_3,

    # G6 시간축·표본
    "time_index": _time_index,
    "batter_pitcher_n_ratio": _batter_pitcher_n_ratio,
    "mix_coverage": _mix_coverage,
    "pitcher_succ_eb": _pitcher_succ_eb,

    # G7 상황 인코딩
    "count_signed": _count_signed,
    "runners_weighted": _runners_weighted,
    "base_out_re": _base_out_re,
    "is_blowout": _is_blowout,
    "inning_x_li": _inning_x_li,

    # G8 구종믹스 × 상황
    "ix_3ball_fastball": _ix_3ball_fastball,
    "ix_2strike_offspeed": _ix_2strike_offspeed,
    "ix_2strike_breaking": _ix_2strike_breaking,
    "mixentropy_x_countpressure": _mixentropy_x_countpressure,
}

# ── 깔때기 스크리닝용 그룹 정의 ─────────────────────────────
# 0단계: V33_ALL 전부 투입 → 피처 축이 살아있는지 1회로 판정
# 1단계: 그룹 단위 → 어느 그룹이 신호를 갖는지
# 2단계: 통과 그룹 내부 개별 (다중비교 부담을 그룹 통과분으로 한정)
V33_GROUPS = {
    "g1_failmode": [
        "residual_fail_rate", "residual_fail_rate_prev5", "middle_share_of_fail",
        "reverse_share_of_fail", "residual_share_of_fail", "fail_mode_entropy",
        "batter_residual_fail",
    ],
    "g2_ballstrike": [
        "inplay_rate", "ball_minus_strike", "ball_strike_ratio",
        "strike_x_middle", "ball_over_fail",
    ],
    "g3_interaction": [
        "ix_2strike_success", "ix_2strike_middle", "ix_2strike_reverse",
        "ix_2strike_strike", "ix_2strike_residual", "ix_must_reverse", "ix_must_strike",
        "ix_countpressure_success", "ix_countpressure_residual",
        "ix_samehand_middle", "ix_samehand_reverse",
        "ix_batter_2strike", "ix_batter_samehand", "ix_batter_risp",
        "ix_li_success", "ix_outs_success",
        "ix_risp_reverse", "ix_risp_residual",
        "ix_highli_success", "ix_highli_middle", "ix_highli_reverse", "ix_highli_residual",
        "ix_lateclose_middle", "ix_lateclose_reverse",
    ],
    "g4_log5": [
        "log5_success", "log5_logit", "log5_middle",
        "pb_success_diff", "pb_success_prod", "pb_mid_diff",
    ],
    "g5_form": [
        "form_trend_1_3", "form_trend_3_5", "form_accel", "form_vol",
        "form_range", "form_delta_abs", "mid_form_delta", "mid_form_trend_1_3",
    ],
    "g6_time_sample": [
        "time_index", "batter_pitcher_n_ratio", "mix_coverage", "pitcher_succ_eb",
    ],
    "g7_situation": [
        "count_signed", "runners_weighted", "base_out_re", "is_blowout", "inning_x_li",
    ],
    "g8_pitchmix": [
        "ix_3ball_fastball", "ix_2strike_offspeed", "ix_2strike_breaking",
        "mixentropy_x_countpressure",
    ],
}

# 전체 후보 (기시도 pit_we/form_delta 는 제외)
V33_ALL = [n for names in V33_GROUPS.values() for n in names]

# log5/EB 계열은 set_league_mean() 선행 필요
NEEDS_LEAGUE_MEAN = {"log5_success", "log5_logit", "log5_middle", "pitcher_succ_eb"}



# ─────────────────────────────────────────────────────────────
# 1차 실행용 26개 (기록 보존 — 기준선으로는 폐기)
# ─────────────────────────────────────────────────────────────
RAW_KEEP = [
    "season", "game_month", "inning",
    "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_pitcher_team",
    "base_state", "li",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_offspeed_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
]
BASE_CAT = ["base_state", "platoon"]
OPTIONAL_CAT = {"pitcher_id", "count_state", "base_out_state"}


def build(df: pd.DataFrame, extras=()) -> pd.DataFrame:
    out = df[RAW_KEEP].copy()
    out["platoon"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
    out["pit_we"] = _pit_we(df)
    out["form_delta"] = _form_delta(df)
    out["pitcher_ahead"] = (df["strikes_before"] > df["balls_before"]).astype("int8")
    out["is_two_strike"] = df["strikes_before"].eq(2).astype("int8")
    out["abs_score_diff"] = df["score_diff_pitcher_team"].abs()

    for e in extras:
        if e == "pitcher_id":
            out["pitcher_id"] = df["pitcher_id"].astype("int32")
        elif e == "asof_batter_n":
            out["asof_batter_n"] = df["asof_batter_n"]
        elif e == "count_state":
            out["count_state"] = df["balls_before"].astype(str) + "_" + df["strikes_before"].astype(str)
        elif e == "base_out_state":
            out["base_out_state"] = df["base_state"].astype(str) + "_" + df["outs_before"].astype(str)
        elif e == "pitchmix_entropy":
            out["pitchmix_entropy"] = _pitchmix_entropy(df)
        else:
            raise ValueError(f"unknown extra: {e}")
    return out


def cat_features(extras=()):
    return BASE_CAT + [e for e in extras if e in OPTIONAL_CAT]


def prepare(x: pd.DataFrame, cats) -> pd.DataFrame:
    """CatBoost 입력 정리 — 범주형은 문자열, 결측은 토큰."""
    x = x.copy()
    for c in cats:
        x[c] = x[c].astype(str).fillna("__NA__").replace("nan", "__NA__")
    return x


# ─────────────────────────────────────────────────────────────
# V24 원본 89개 (기준선)
# ─────────────────────────────────────────────────────────────
V24_RAW = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
V24_CAT = ["top_bottom", "game_type", "base_state",
           "count_state", "base_out_state", "platoon"]

# 자주 쓰는 삭제 묶음
IDS4 = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]


def batter_shrink_prior(fit_df: pd.DataFrame) -> dict:
    """FIT셋에서만 계산하는 경험적 베이즈 사전평균. valid/test에는 이 값을 그대로 씌운다
    (검증셋을 보고 사전을 다시 잡으면 규칙 5절 위반 소지가 있어 fit 전용으로 고정)."""
    return {
        "succ": float(fit_df["asof_batter_success_rate"].mean()),
        "mid": float(fit_df["asof_batter_middle_rate"].mean()),
    }


def _batter_shrink(df: pd.DataFrame, prior: dict, k: float) -> pd.DataFrame:
    """asof_batter_n(표본수)을 이용한 베타-이항 부분풀링(경험적 베이즈) 축소추정치.

    shrunk = (n*rate + k*prior) / (n + k)
    n이 작을수록(신인/소표본 타자) prior 쪽으로 당겨지고, n이 크면 raw rate에 수렴한다.
    raw n 자체는 피처로 노출하지 않는다(run_screen 스크리닝에서 raw n 노출은 -14.3으로 손해 확인됨).
    """
    n = df["asof_batter_n"].fillna(0.0)
    succ = df["asof_batter_success_rate"].fillna(prior["succ"])
    mid = df["asof_batter_middle_rate"].fillna(prior["mid"])
    out = pd.DataFrame(index=df.index)
    out["batter_succ_shrunk"] = (n * succ + k * prior["succ"]) / (n + k)
    out["batter_mid_shrunk"] = (n * mid + k * prior["mid"]) / (n + k)
    return out


def build_v24(df: pd.DataFrame, add=(), drop=(), trackman=None, min_margin=0.0,
              batter_shrink=None) -> pd.DataFrame:
    """V24 89개 + add 파생 + drop 제거 + Trackman 물리 피처(선택) + 타자 shrinkage(선택).

    trackman: load_trackman() 결과. min_margin 미만인 투수는 결측 처리한다
    (매칭 확신이 낮은 투수에게 엉뚱한 물리량이 붙는 것을 막음. CatBoost 는 결측을 네이티브 처리).
    batter_shrink: (prior_dict, k) 튜플. prior_dict는 batter_shrink_prior(fit_df) 결과를
    fit에서 한 번만 계산해 fit/valid 양쪽에 동일하게 씌운다."""
    d = df[V24_RAW].copy()
    balls, strikes = df["balls_before"], df["strikes_before"]
    outs, sd = df["outs_before"], df["score_diff_pitcher_team"]

    d["count_state"] = balls.astype(str) + "_" + strikes.astype(str)
    d["base_out_state"] = df["base_state"].astype(str) + "_" + outs.astype(str)
    d["platoon"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)

    d["is_full_count"] = ((balls == 3) & (strikes == 2)).astype("int8")
    d["is_two_strike"] = (strikes == 2).astype("int8")
    d["is_three_ball"] = (balls == 3).astype("int8")
    d["pitcher_ahead"] = (strikes > balls).astype("int8")
    d["pitcher_behind"] = (balls > strikes).astype("int8")
    d["count_pressure"] = (balls + strikes).astype("int8")

    d["is_late_inning"] = (df["inning"] >= 7).astype("int8")
    d["is_extra_inning"] = (df["inning"] >= 10).astype("int8")
    d["abs_score_diff"] = sd.abs()
    d["is_tie_game"] = (sd == 0).astype("int8")
    d["is_close_game"] = (sd.abs() <= 1).astype("int8")
    d["pitcher_team_leading"] = (sd > 0).astype("int8")
    d["pitcher_team_trailing"] = (sd < 0).astype("int8")

    d["has_runner"] = (df["num_runners_on"] > 0).astype("int8")
    d["is_scoring_position"] = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype("int8")
    d["runner_pressure"] = df["num_runners_on"] * (outs + 1)
    d["late_close"] = (d["is_late_inning"] & d["is_close_game"]).astype("int8")
    d["winexp_gap_home_away"] = df["home_win_expectancy"] - df["away_win_expectancy"]
    d["same_hand_matchup"] = (df["pitcher_hand"].astype(str) == df["batter_hand"].astype(str)).astype("int8")
    d["teams_same"] = (df["pitcher_team_id"] == df["batter_team_id"]).astype("int8")

    p_succ = df["asof_pitcher_success_rate"]; p_mid = df["asof_pitcher_middle_rate"]
    p_ball = df["asof_pitcher_ball_rate"];    p_rev = df["asof_pitcher_reverse_rate"]
    p_prev5 = df["asof_pitcher_prev5_game_success_rate"]

    norm = _pitchmix_norm(df)
    d["pitchmix_entropy"] = -(norm * np.log(norm.replace(0, np.nan))).sum(axis=1).fillna(0)
    d["pitchmix_max_rate"] = norm.max(axis=1)
    d["fastball_minus_breaking"] = df[MIXCOLS[0]] - df[MIXCOLS[1]]
    d["fastball_minus_offspeed"] = df[MIXCOLS[0]] - df[MIXCOLS[2]]
    d["breaking_minus_offspeed"] = df[MIXCOLS[1]] - df[MIXCOLS[2]]
    d["is_fastball_heavy"] = (df[MIXCOLS[0]] >= 0.6).astype("int8")
    d["is_balanced_mix"] = (d["pitchmix_max_rate"] <= 0.5).astype("int8")

    must = (balls == 3).astype("int8")
    waste = ((strikes == 2) & (balls <= 1)).astype("int8")
    loaded = ((df["runner_on_1b"] == 1) & (df["runner_on_2b"] == 1) & (df["runner_on_3b"] == 1)).astype("int8")
    high_li = (df["li"] >= 1.5).astype("int8")

    d["ix_must_ball"] = must * p_ball
    d["ix_must_middle"] = must * p_mid
    d["ix_must_success"] = must * p_succ
    d["ix_waste_ball"] = waste * p_ball
    d["ix_waste_reverse"] = waste * p_rev
    d["ix_risp_success"] = d["is_scoring_position"] * p_succ
    d["ix_risp_middle"] = d["is_scoring_position"] * p_mid
    d["ix_loaded_ball"] = loaded * p_ball
    d["ix_loaded_middle"] = loaded * p_mid
    d["ix_highli_prev5"] = high_li * p_prev5
    d["ix_lateclose_success"] = d["late_close"] * p_succ
    d["ix_samehand_success"] = d["same_hand_matchup"] * p_succ

    if add:
        bad = [a for a in add if a not in NEW_DERIVED]
        if bad:
            raise ValueError(f"unknown add: {bad}")
        # 63개를 하나씩 insert 하면 DataFrame 이 심하게 fragment 되어 느려진다 → 한 번에 concat
        extra = pd.DataFrame({a: NEW_DERIVED[a](df) for a in add}, index=d.index)
        d = pd.concat([d, extra], axis=1)

    if batter_shrink is not None:
        prior, k = batter_shrink
        d = d.join(_batter_shrink(df, prior, k))

    if drop:
        missing = [c for c in drop if c not in d.columns]
        if missing:
            raise ValueError(f"drop 대상이 없습니다: {missing}")
        d = d.drop(columns=list(drop))

    if trackman is not None:
        tf = trackman[trackman["tm_match_margin"] >= min_margin] \
                     .drop(columns=["tm_match_cost", "tm_match_margin"])
        d = d.join(tf, on="pitcher_id")
    return d


def v24_cat_features(drop=()):
    return [c for c in V24_CAT if c not in set(drop)]


# ══════════════════════════════════════════════════════════════════════════
# V33 추론부. 위쪽은 features.py 원본이 그대로 인라인된 것이며,
# make_submission_v33.py 가 [features.py + 이 템플릿] 을 이어붙여 script.py 를 생성한다.
#
# 왜 인라인인가: 피처가 63개 늘어난 상태에서 학습용/추론용 이중 구현을 두면 불일치 위험이
# 크다. 그렇다고 features.py 를 zip 에 따로 넣으면 944 를 받은 V32 zip 과 파일 구성이
# 달라진다. 인라인하면 단일 정본(features.py)을 유지하면서 zip 구조는 V32 와 동일해진다.
# ══════════════════════════════════════════════════════════════════════════
import os
import warnings

import joblib

ID_COL = "row_id"
TARGET_COL = "control_success"
EPS_SUBMIT = 1e-6

# 폴백 상수 (cats.pkl 에 값이 있으면 그쪽이 우선)
CAT_WEIGHT = 0.6
LOGIT_SCALE = 1.09
LOGIT_DELTA = -0.0521

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def _model_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "model", filename),
        os.path.join(script_dir, filename),
        os.path.join("./model", filename),
        "./" + filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Model file not found for %s. Tried: %s" % (filename, candidates))


def _logit(p):
    p = np.clip(p, EPS_SUBMIT, 1.0 - EPS_SUBMIT)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    data_dir, out_dir = "./data", "./output"
    test_path = os.path.join(data_dir, "test.csv")
    sample_path = os.path.join(data_dir, "sample_submission.csv")
    out_path = os.path.join(out_dir, "submission.csv")

    cat_artifact = joblib.load(_model_path("cats.pkl"))
    lx_artifact = joblib.load(_model_path("lx5.pkl"))

    features = cat_artifact["features"]
    cat_features = cat_artifact["cat_features"]
    added = tuple(cat_artifact.get("v33_added", ()))
    cat_weight = float(cat_artifact.get("cat_weight", CAT_WEIGHT))
    scale = float(cat_artifact.get("logit_scale", LOGIT_SCALE))
    delta = float(cat_artifact.get("logit_delta", LOGIT_DELTA))

    league_mean = cat_artifact.get("league_mean")
    if league_mean is None:
        raise ValueError("cats.pkl 에 league_mean 이 없습니다 (log5/EB 피처에 필요)")
    set_league_mean(float(league_mean))     # 학습 데이터에서 산출된 상수 — test 미참조

    test = pd.read_csv(test_path, encoding="utf-8-sig")
    sub = pd.read_csv(sample_path, encoding="utf-8-sig")

    # drop 없이 한 번만 생성 → CatBoost(151)와 lx5(V24 89)가 같은 프레임에서 각자 고른다.
    # (lx5 는 asof_batter_n 을 쓰는데 CatBoost 쪽은 그걸 뺀 구성이라 여기서 drop 을 걸면 안 됨)
    fe = build_v24(test, add=added, drop=())

    missing = [c for c in features if c not in fe.columns]
    if missing:
        raise ValueError("CatBoost 피처 누락: %s" % missing[:10])
    missing_lx = [c for c in lx_artifact["features"] if c not in fe.columns]
    if missing_lx:
        raise ValueError("lx5 피처 누락: %s" % missing_lx[:10])

    cat_x = prepare(fe[features].copy(), [c for c in cat_features if c in features])
    pred_cat = np.mean([m.predict_proba(cat_x)[:, 1] for _, m in cat_artifact["models"]], axis=0)

    lx_x = fe[lx_artifact["features"]]
    pred_lx = np.mean([m.predict_proba(lx_x)[:, 1] for _, m in lx_artifact["models"]], axis=0)

    raw_blend = cat_weight * pred_cat + (1.0 - cat_weight) * pred_lx
    preds = _sigmoid(scale * _logit(raw_blend) + delta)

    pred_df = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET_COL: preds})
    sub = sub[[ID_COL]].merge(pred_df, on=ID_COL, how="left")
    if sub[TARGET_COL].isna().any():
        raise ValueError("Some sample_submission row_id values were not found in test.csv")

    sub[TARGET_COL] = sub[TARGET_COL].clip(0.0, 1.0)
    os.makedirs(out_dir, exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved: %s rows=%d feats=%d added=%d" % (out_path, len(sub), len(features), len(added)))


if __name__ == "__main__":
    main()

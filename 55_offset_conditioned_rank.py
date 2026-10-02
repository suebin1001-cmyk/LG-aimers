"""
Offset-conditioned rank score 실험.

핵심 질문: "burst onset transient(offset=0에서 체계적으로 높은 MSE)를 architecture가 아니라
detection score의 position-conditioned calibration으로 해결할 수 있는가?"

비교 5개 config (backbone은 CNN/TCN 유지, score만 바꿈):
  A. CNN + mode-conditioned mean/std z-score      (기존 baseline, mode(3-cluster) 조건)
  B. CNN + global rank  (percentile | 전체 TRAIN 정상 reference, 조건 없음)
  C. CNN + offset-conditioned rank (percentile | 동일 offset bin의 TRAIN 정상 reference)
  D. TCN + global rank
  E. TCN + offset-conditioned rank

offset bin (burst-relative window offset, train 분포 기준 BIN 경계는 사전에 고정, 데이터로 재선택하지 않음):
  B0=0, B1=1~2, B2=3~4, B3=5~9, B4=10+

reference는 전부 TRAIN 정상만 사용. threshold(zt)/CUSUM(k,h)는 VALIDATION에서만 선택,
TEST는 그 고정된 설정으로 1회만 평가(exploratory). CNN과 TCN은 seed당 1회만 학습하고
그 reconstruction MSE로 여러 score를 유도한다(동일 모델, 다른 채점 방식).

추가로 반드시 확인:
  - 정상 offset=0 윈도우가 FP로 플래그되는 수
  - 실제 anomaly 중 offset=0 윈도우의 detection recall
  - anomaly burst가 offset=0에서 최초 탐지되는 비율과 그때의 lead-time
"""
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics

use_col = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
WINDOW = 15
CORR_WINDOW = 30
SEEDS = [42, 142, 242, 342, 442]

def assign_burst_id(df, gap_mult=3.0):
    ts = pd.to_datetime(df["TimeStamp"])
    diffs = ts.diff().dt.total_seconds()
    med = diffs.median()
    is_new_burst = (diffs > med * gap_mult) | diffs.isna()
    return is_new_burst.cumsum().values - 1

normal_raw = pd.read_csv("normal_with_phase.csv", index_col=0)
outlier_raw = pd.read_csv("outlier_with_phase.csv", index_col=0)
normal_raw["burst_id"] = assign_burst_id(normal_raw)
outlier_raw["burst_id"] = assign_burst_id(outlier_raw)

X_norm_modes = StandardScaler().fit_transform(normal_raw[use_col].abs())
km_mode = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X_norm_modes)
normal_raw["mode"] = km_mode.labels_
outlier_raw["mode"] = km_mode.predict(StandardScaler().fit(normal_raw[use_col].abs()).transform(outlier_raw[use_col].abs()))

normal_p = normal_raw.copy(); normal_p[use_col] = normal_p[use_col].apply(lambda s: s.abs())
scaler = MinMaxScaler().fit(normal_p[use_col])
normal_scaled3 = scaler.transform(normal_p[use_col])
outlier_p = outlier_raw.copy(); outlier_p[use_col] = outlier_p[use_col].apply(lambda s: s.abs())
outlier_scaled3 = scaler.transform(outlier_p[use_col])

def rolling_corr_all(df, corr_window):
    out = pd.Series(index=df.index, dtype=float)
    for bid, g in df.groupby("burst_id"):
        c = g["AI0_Vibration"].rolling(corr_window, min_periods=3).corr(g["AI1_Vibration"])
        out.loc[g.index] = c
    return out.fillna(0.0).values

normal_raw["phase_corr_row"] = rolling_corr_all(normal_raw, CORR_WINDOW)
outlier_raw["phase_corr_row"] = rolling_corr_all(outlier_raw, CORR_WINDOW)
normal_corr_scaled = (normal_raw["phase_corr_row"].values + 1) / 2
outlier_corr_scaled = (outlier_raw["phase_corr_row"].values + 1) / 2
normal_scaled4 = np.concatenate([normal_scaled3, normal_corr_scaled.reshape(-1, 1)], axis=1)
outlier_scaled4 = np.concatenate([outlier_scaled3, outlier_corr_scaled.reshape(-1, 1)], axis=1)

def burst_starts(burst, window):
    n = len(burst) - window + 1
    return np.array([i for i in range(n) if burst[i] == burst[i + window - 1]])

def group_split(burst_ids, unique_bursts, train_frac, seed):
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_bursts)
    n_train = int(len(shuffled) * train_frac)
    return set(shuffled[:n_train]), set(shuffled[n_train:])

def build_cnn_ae(window, n_features=4):
    inp = layers.Input(shape=(window, n_features))
    x = layers.Conv1D(16, 5, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Conv1D(8, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(3, padding="same")(x)
    x = layers.Conv1D(8, 5, padding="same", activation="relu")(x)
    x = layers.UpSampling1D(3)(x)
    x = layers.Conv1D(16, 5, padding="same", activation="relu")(x)
    x = layers.UpSampling1D(2)(x)
    out = layers.Conv1D(n_features, 5, padding="same", activation=None)(x)
    out = layers.Lambda(lambda t: t[:, :window, :])(out)
    m = models.Model(inp, out); m.compile(optimizer="adam", loss="mse")
    return m

def build_tcn_ae_window(window=15, n_features=4):
    inp = layers.Input(shape=(window, n_features))
    x = inp
    for d in [1, 2, 4]:
        x = layers.Conv1D(16, 3, padding="causal", dilation_rate=d, activation="relu")(x)
    x = layers.Conv1D(8, 3, padding="causal", dilation_rate=1, activation="relu", name="bottleneck")(x)
    for d in [4, 2, 1]:
        x = layers.Conv1D(16, 3, padding="causal", dilation_rate=d, activation="relu")(x)
    out = layers.Conv1D(n_features, 3, padding="causal", dilation_rate=1, activation=None)(x)
    m = models.Model(inp, out); m.compile(optimizer="adam", loss="mse")
    return m

def recon_error_window(m, X, batch=4096):
    errs = []
    for i in range(0, len(X), batch):
        pred = m.predict(X[i:i+batch], verbose=0)
        errs.append(np.mean(np.power(X[i:i+batch] - pred, 2), axis=(1, 2)))
    return np.concatenate(errs)

def mode_zscore_meanstd(vals, mode_vals, stats):
    zz = np.zeros(len(vals))
    for md in [0, 1, 2]:
        mask = mode_vals == md
        mmean, mstd = stats[md]
        zz[mask] = (vals[mask] - mmean) / (mstd + 1e-9)
    return zz

def global_rank_score(mse_train, mse_vals):
    train_sorted = np.sort(mse_train)
    n = len(train_sorted)
    return np.searchsorted(train_sorted, mse_vals, side="right") / n

def offset_bin_of(off):
    off = np.asarray(off)
    bins = np.zeros(len(off), dtype=int)
    bins[off == 0] = 0
    bins[(off >= 1) & (off <= 2)] = 1
    bins[(off >= 3) & (off <= 4)] = 2
    bins[(off >= 5) & (off <= 9)] = 3
    bins[off >= 10] = 4
    return bins

def offset_cond_rank_score(mse_train, bin_train, mse_vals, bin_vals):
    score = np.zeros(len(mse_vals))
    for b in range(5):
        train_vals = np.sort(mse_train[bin_train == b])
        if len(train_vals) < 5:
            train_vals = np.sort(mse_train)  # 해당 bin에 train sample이 너무 적으면 전체로 fallback
        n = len(train_vals)
        mv = bin_vals == b
        if mv.sum() == 0: continue
        score[mv] = np.searchsorted(train_vals, mse_vals[mv], side="right") / n
    return score

def cusum_flags(z, burst_ids, starts, k, h):
    order = np.lexsort((starts, burst_ids))
    z_ord = z[order]; burst_ord = burst_ids[order]
    S = np.zeros(len(z_ord)); flag = np.zeros(len(z_ord), dtype=int)
    prev_burst = None; s_val = 0.0
    for i in range(len(z_ord)):
        if burst_ord[i] != prev_burst:
            s_val = 0.0; prev_burst = burst_ord[i]
        s_val = max(0.0, s_val + (z_ord[i] - k))
        S[i] = s_val
        flag[i] = 1 if s_val > h else 0
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    return flag[inv], S[inv]

def metrics_of(pred, Y):
    tp = int(((pred==1)&(Y==1)).sum()); fn = int(((pred==0)&(Y==1)).sum())
    fp = int(((pred==1)&(Y==0)).sum()); tn = int(((pred==0)&(Y==0)).sum())
    f1 = metrics.f1_score(Y, pred)
    recall = tp/(tp+fn) if (tp+fn) else float("nan")
    fpr = fp/(fp+tn) if (fp+tn) else float("nan")
    return dict(f1=f1, recall=recall, fpr=fpr, tp=tp, fn=fn, fp=fp, tn=tn)

def low_sev_recall_of(pred, Y, sev, sev_thr):
    mask = (Y == 1) & (sev < sev_thr)
    return (pred[mask] == 1).mean() if mask.sum() > 0 else float("nan")

def select_on_valid(score_valid, Y_valid, z_candidates, k_candidates, h_candidates,
                     burst_valid_all, start_valid_all):
    best_zt, best_f1 = None, -1
    for zt in z_candidates:
        f1 = metrics.f1_score(Y_valid, (score_valid > zt).astype(int))
        if f1 > best_f1 + 1e-12: best_zt, best_f1 = zt, f1
    pred_state_valid = (score_valid > best_zt).astype(int)
    best = (None, None, -1)
    for k in k_candidates:
        for h in h_candidates:
            flag_v, _ = cusum_flags(score_valid, burst_valid_all, start_valid_all, k, h)
            pred = (pred_state_valid | flag_v).astype(int)
            f1 = metrics.f1_score(Y_valid, pred)
            if f1 > best[2] + 1e-12: best = (k, h, f1)
    best_k, best_h, _ = best
    return dict(zt=best_zt, k=best_k, h=best_h, val_f1=best[2])

def apply_fixed(score, zt, k, h, burst_all, start_all):
    pred_state = (score > zt).astype(int)
    flag, _ = cusum_flags(score, burst_all, start_all, k, h)
    return (pred_state | flag).astype(int)

def event_leadtime_detail(burst_ids_u, offsets_u, pred_u, Y_u, burst_dur_lookup, stride=0.1):
    """burst별 (최초 탐지 offset(윈도우 단위), lead_time) 반환"""
    out = []
    for bid in np.unique(burst_ids_u[Y_u == 1]):
        mask = (burst_ids_u == bid)
        offs = offsets_u[mask]; pr = pred_u[mask]
        order = np.argsort(offs)
        pr_sorted = pr[order]; offs_sorted = offs[order]
        fired = np.where(pr_sorted == 1)[0]
        if len(fired) > 0:
            first_off = int(offs_sorted[fired[0]])
            detect_t = first_off * stride
            dur = burst_dur_lookup.get(bid, offs_sorted[-1] * stride)
            lead = max(0.0, dur - detect_t)
            out.append((bid, first_off, lead))
        else:
            out.append((bid, None, None))
    return out

def get_splits(seed_base):
    start_n = burst_starts(normal_raw["burst_id"].values, WINDOW)
    burst_n_at_start = normal_raw["burst_id"].values[start_n]
    uniq_n = np.unique(burst_n_at_start)
    train_b, holdout_b = group_split(burst_n_at_start, uniq_n, 0.8, seed=seed_base)
    holdout_mask = np.isin(burst_n_at_start, list(holdout_b))
    holdout_bursts = np.unique(burst_n_at_start[holdout_mask])
    valid_b, test_b = group_split(burst_n_at_start[holdout_mask], holdout_bursts, 0.5, seed=seed_base+1)
    start_o = burst_starts(outlier_raw["burst_id"].values, WINDOW)
    burst_o_at_start = outlier_raw["burst_id"].values[start_o]
    uniq_o = np.unique(burst_o_at_start)
    ovalid_b, otest_b = group_split(burst_o_at_start, uniq_o, 0.5, seed=seed_base+2)
    return dict(train_b=train_b, valid_b=valid_b, test_b=test_b, ovalid_b=ovalid_b, otest_b=otest_b,
                start_n=start_n, burst_n_at_start=burst_n_at_start,
                start_o=start_o, burst_o_at_start=burst_o_at_start)

def offset_within_burst(burst_at_start, start_at_start):
    off = np.zeros(len(start_at_start), dtype=int)
    for bid in np.unique(burst_at_start):
        m = burst_at_start == bid
        off[m] = start_at_start[m] - start_at_start[m].min()
    return off

def prepare_arrays(seed_base, splits):
    start_n = splits["start_n"]; burst_n_at_start = splits["burst_n_at_start"]
    start_o = splits["start_o"]; burst_o_at_start = splits["burst_o_at_start"]
    train_b, valid_b, test_b = splits["train_b"], splits["valid_b"], splits["test_b"]
    ovalid_b, otest_b = splits["ovalid_b"], splits["otest_b"]

    Xn = np.array([normal_scaled4[s:s+WINDOW] for s in start_n])
    Xo = np.array([outlier_scaled4[s:s+WINDOW] for s in start_o])
    mode_n_at_start = normal_raw["mode"].values[start_n + WINDOW - 1]
    mode_o_at_start = outlier_raw["mode"].values[start_o + WINDOW - 1]
    off_n_all = offset_within_burst(burst_n_at_start, start_n)
    off_o_all = offset_within_burst(burst_o_at_start, start_o)

    train_mask = np.isin(burst_n_at_start, list(train_b))
    nvalid_mask = np.isin(burst_n_at_start, list(valid_b))
    ntest_mask = np.isin(burst_n_at_start, list(test_b))
    ovalid_mask = np.isin(burst_o_at_start, list(ovalid_b))
    otest_mask = np.isin(burst_o_at_start, list(otest_b))

    X_train = Xn[train_mask]; mode_train = mode_n_at_start[train_mask]; off_train = off_n_all[train_mask]
    bin_train = offset_bin_of(off_train)

    X_valid = np.concatenate([Xn[nvalid_mask], Xo[ovalid_mask]])
    Y_valid = np.concatenate([np.zeros(nvalid_mask.sum()), np.ones(ovalid_mask.sum())])
    mode_valid = np.concatenate([mode_n_at_start[nvalid_mask], mode_o_at_start[ovalid_mask]])
    off_valid = np.concatenate([off_n_all[nvalid_mask], off_o_all[ovalid_mask]])
    bin_valid = offset_bin_of(off_valid)

    X_test = np.concatenate([Xn[ntest_mask], Xo[otest_mask]])
    Y_test = np.concatenate([np.zeros(ntest_mask.sum()), np.ones(otest_mask.sum())])
    mode_test = np.concatenate([mode_n_at_start[ntest_mask], mode_o_at_start[otest_mask]])
    off_test = np.concatenate([off_n_all[ntest_mask], off_o_all[otest_mask]])
    bin_test = offset_bin_of(off_test)

    burst_valid_all = np.concatenate([burst_n_at_start[nvalid_mask], burst_o_at_start[ovalid_mask] + 100000])
    start_valid_all = np.concatenate([start_n[nvalid_mask], start_o[ovalid_mask]])
    burst_test_all = np.concatenate([burst_n_at_start[ntest_mask], burst_o_at_start[otest_mask] + 100000])
    start_test_all = np.concatenate([start_n[ntest_mask], start_o[otest_mask]])

    sev_mu = normal_raw.loc[normal_raw["burst_id"].isin(train_b), use_col].mean()
    sev_sigma = normal_raw.loc[normal_raw["burst_id"].isin(train_b), use_col].std()
    def window_severity(raw_df, starts, window):
        Z = (raw_df[use_col] - sev_mu) / (sev_sigma + 1e-9)
        row_sev = np.sqrt((Z.values ** 2).sum(axis=1))
        out = np.zeros(len(starts))
        for i, s in enumerate(starts):
            out[i] = row_sev[s:s+window].mean()
        return out
    sev_test = np.concatenate([window_severity(normal_raw, start_n[ntest_mask], WINDOW),
                                window_severity(outlier_raw, start_o[otest_mask], WINDOW)])

    def burst_dur_lookup_for(burst_ids_all):
        d = {}
        for bid in np.unique(burst_ids_all):
            raw_bid = int(bid if bid < 100000 else bid - 100000)
            src = normal_raw if bid < 100000 else outlier_raw
            L = int((src["burst_id"] == raw_bid).sum())
            d[bid] = L * 0.1
        return d
    burst_dur_test = burst_dur_lookup_for(burst_test_all)

    return dict(X_train=X_train, mode_train=mode_train, bin_train=bin_train,
                X_valid=X_valid, Y_valid=Y_valid, mode_valid=mode_valid, bin_valid=bin_valid,
                burst_valid_all=burst_valid_all, start_valid_all=start_valid_all,
                X_test=X_test, Y_test=Y_test, mode_test=mode_test, bin_test=bin_test, off_test=off_test,
                burst_test_all=burst_test_all, start_test_all=start_test_all, sev_test=sev_test,
                burst_dur_test=burst_dur_test)

Z_RANK = np.linspace(0.5, 1.5, 60)
Z_MEANSTD = np.linspace(0.5, 8.0, 60)
K_C = np.linspace(0.0, 2.0, 9); H_C = np.linspace(1.0, 40.0, 14)

def full_eval(score_valid, score_test, zgrid, arrs):
    sel = select_on_valid(score_valid, arrs["Y_valid"], zgrid, K_C, H_C,
                           arrs["burst_valid_all"], arrs["start_valid_all"])
    pred_test = apply_fixed(score_test, sel["zt"], sel["k"], sel["h"],
                             arrs["burst_test_all"], arrs["start_test_all"])
    m = metrics_of(pred_test, arrs["Y_test"])
    m["lowsev"] = low_sev_recall_of(pred_test, arrs["Y_test"], arrs["sev_test"], 3.0)

    # offset=0 전용 지표
    off0_mask = arrs["off_test"] == 0
    normal_off0_mask = off0_mask & (arrs["Y_test"] == 0)
    anomaly_off0_mask = off0_mask & (arrs["Y_test"] == 1)
    m["normal_off0_fp"] = int((pred_test[normal_off0_mask] == 1).sum())
    m["normal_off0_total"] = int(normal_off0_mask.sum())
    m["anomaly_off0_recall"] = float((pred_test[anomaly_off0_mask] == 1).mean()) if anomaly_off0_mask.sum() > 0 else float("nan")
    m["anomaly_off0_total"] = int(anomaly_off0_mask.sum())

    detail = event_leadtime_detail(arrs["burst_test_all"], arrs["off_test"], pred_test, arrs["Y_test"], arrs["burst_dur_test"])
    leads_all = [d[2] for d in detail if d[2] is not None]
    m["lead"] = float(np.mean(leads_all)) if leads_all else 0.0
    m["detected"] = sum(1 for d in detail if d[1] is not None); m["total_anom_burst"] = len(detail)
    off0_detected = [d for d in detail if d[1] == 0]
    m["frac_first_detect_at_off0"] = len(off0_detected) / len(detail) if len(detail) else float("nan")
    m["lead_when_first_at_off0"] = float(np.mean([d[2] for d in off0_detected])) if off0_detected else float("nan")

    fpmask = (pred_test == 1) & (arrs["Y_test"] == 0)
    m["fp_bursts"] = sorted(set(arrs["burst_test_all"][fpmask].tolist()))
    return m

configs = ["A_CNN_meanstd", "B_CNN_globalrank", "C_CNN_offsetrank", "D_TCN_globalrank", "E_TCN_offsetrank"]
all_results = {c: [] for c in configs}

for seed_base in SEEDS:
    print(f"\n{'='*100}\nseed={seed_base}\n{'='*100}")
    splits = get_splits(seed_base)
    arrs = prepare_arrays(seed_base, splits)

    # ---- CNN (1회 학습) ----
    keras.utils.set_random_seed(seed_base)
    cnn = build_cnn_ae(WINDOW, 4)
    es = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    cnn.fit(arrs["X_train"], arrs["X_train"], epochs=150, batch_size=64,
            validation_split=0.1, callbacks=[es], verbose=0)
    mse_train_c = recon_error_window(cnn, arrs["X_train"])
    mse_valid_c = recon_error_window(cnn, arrs["X_valid"])
    mse_test_c = recon_error_window(cnn, arrs["X_test"])

    stats_c = {md: (mse_train_c[arrs["mode_train"]==md].mean(), mse_train_c[arrs["mode_train"]==md].std()) for md in [0,1,2]}
    A_valid = mode_zscore_meanstd(mse_valid_c, arrs["mode_valid"], stats_c)
    A_test = mode_zscore_meanstd(mse_test_c, arrs["mode_test"], stats_c)
    m = full_eval(A_valid, A_test, Z_MEANSTD, arrs); all_results["A_CNN_meanstd"].append(m)
    print(f"  A_CNN_meanstd    : F1={m['f1']:.4f} FPR={m['fpr']:.5f} FP={m['fp']} FN={m['fn']} LowSev={m['lowsev']:.4f} lead={m['lead']:.2f}s | "
          f"normal_off0_FP={m['normal_off0_fp']}/{m['normal_off0_total']} anomaly_off0_recall={m['anomaly_off0_recall']:.3f} "
          f"frac_first@off0={m['frac_first_detect_at_off0']:.2f}")

    B_valid = global_rank_score(mse_train_c, mse_valid_c)
    B_test = global_rank_score(mse_train_c, mse_test_c)
    m = full_eval(B_valid, B_test, Z_RANK, arrs); all_results["B_CNN_globalrank"].append(m)
    print(f"  B_CNN_globalrank : F1={m['f1']:.4f} FPR={m['fpr']:.5f} FP={m['fp']} FN={m['fn']} LowSev={m['lowsev']:.4f} lead={m['lead']:.2f}s | "
          f"normal_off0_FP={m['normal_off0_fp']}/{m['normal_off0_total']} anomaly_off0_recall={m['anomaly_off0_recall']:.3f} "
          f"frac_first@off0={m['frac_first_detect_at_off0']:.2f}")

    C_valid = offset_cond_rank_score(mse_train_c, arrs["bin_train"], mse_valid_c, arrs["bin_valid"])
    C_test = offset_cond_rank_score(mse_train_c, arrs["bin_train"], mse_test_c, arrs["bin_test"])
    m = full_eval(C_valid, C_test, Z_RANK, arrs); all_results["C_CNN_offsetrank"].append(m)
    print(f"  C_CNN_offsetrank : F1={m['f1']:.4f} FPR={m['fpr']:.5f} FP={m['fp']} FN={m['fn']} LowSev={m['lowsev']:.4f} lead={m['lead']:.2f}s | "
          f"normal_off0_FP={m['normal_off0_fp']}/{m['normal_off0_total']} anomaly_off0_recall={m['anomaly_off0_recall']:.3f} "
          f"frac_first@off0={m['frac_first_detect_at_off0']:.2f}")

    # ---- TCN (1회 학습) ----
    keras.utils.set_random_seed(seed_base)
    tcn = build_tcn_ae_window(WINDOW, 4)
    es2 = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    tcn.fit(arrs["X_train"], arrs["X_train"], epochs=150, batch_size=64,
            validation_split=0.1, callbacks=[es2], verbose=0)
    mse_train_t = recon_error_window(tcn, arrs["X_train"])
    mse_valid_t = recon_error_window(tcn, arrs["X_valid"])
    mse_test_t = recon_error_window(tcn, arrs["X_test"])

    D_valid = global_rank_score(mse_train_t, mse_valid_t)
    D_test = global_rank_score(mse_train_t, mse_test_t)
    m = full_eval(D_valid, D_test, Z_RANK, arrs); all_results["D_TCN_globalrank"].append(m)
    print(f"  D_TCN_globalrank : F1={m['f1']:.4f} FPR={m['fpr']:.5f} FP={m['fp']} FN={m['fn']} LowSev={m['lowsev']:.4f} lead={m['lead']:.2f}s | "
          f"normal_off0_FP={m['normal_off0_fp']}/{m['normal_off0_total']} anomaly_off0_recall={m['anomaly_off0_recall']:.3f} "
          f"frac_first@off0={m['frac_first_detect_at_off0']:.2f}")

    E_valid = offset_cond_rank_score(mse_train_t, arrs["bin_train"], mse_valid_t, arrs["bin_valid"])
    E_test = offset_cond_rank_score(mse_train_t, arrs["bin_train"], mse_test_t, arrs["bin_test"])
    m = full_eval(E_valid, E_test, Z_RANK, arrs); all_results["E_TCN_offsetrank"].append(m)
    print(f"  E_TCN_offsetrank : F1={m['f1']:.4f} FPR={m['fpr']:.5f} FP={m['fp']} FN={m['fn']} LowSev={m['lowsev']:.4f} lead={m['lead']:.2f}s | "
          f"normal_off0_FP={m['normal_off0_fp']}/{m['normal_off0_total']} anomaly_off0_recall={m['anomaly_off0_recall']:.3f} "
          f"frac_first@off0={m['frac_first_detect_at_off0']:.2f}")

print("\n" + "="*100)
print("5-seed 요약")
print("="*100)
def summarize(name, rows):
    f1 = np.array([r["f1"] for r in rows]); rec = np.array([r["recall"] for r in rows])
    fpr = np.array([r["fpr"] for r in rows]); fp = np.array([r["fp"] for r in rows])
    fn = np.array([r["fn"] for r in rows]); ls = np.array([r["lowsev"] for r in rows])
    lead = np.array([r["lead"] for r in rows])
    noff0fp = np.array([r["normal_off0_fp"] for r in rows]); noff0tot = np.array([r["normal_off0_total"] for r in rows])
    aoff0rec = np.array([r["anomaly_off0_recall"] for r in rows])
    fracfirst = np.array([r["frac_first_detect_at_off0"] for r in rows])
    b67 = sum(67 in r["fp_bursts"] for r in rows); b386 = sum(386 in r["fp_bursts"] for r in rows)
    print(f"[{name}]")
    print(f"  F1={f1.mean():.4f}±{f1.std():.4f} Recall={rec.mean():.4f}±{rec.std():.4f} FPR={fpr.mean():.4f}±{fpr.std():.4f} "
          f"FP={fp.mean():.1f}±{fp.std():.1f} FN={fn.mean():.1f}±{fn.std():.1f} LowSev={ls.mean():.3f}±{ls.std():.3f} lead={lead.mean():.2f}±{lead.std():.2f}s")
    print(f"  normal_off0_FP={noff0fp.mean():.1f}±{noff0fp.std():.1f} (평균 {noff0tot.mean():.0f}개 중) "
          f"anomaly_off0_recall={np.nanmean(aoff0rec):.3f}±{np.nanstd(aoff0rec):.3f} "
          f"frac_first_detect@off0={np.nanmean(fracfirst):.3f} burst67_FP={b67}/5 burst386_FP={b386}/5")
    print(f"  seed별 raw: " + "; ".join([f"s{SEEDS[i]}:F1={rows[i]['f1']:.3f},FP={rows[i]['fp']},off0FP={rows[i]['normal_off0_fp']},off0rec={rows[i]['anomaly_off0_recall']:.2f}" for i in range(len(rows))]))

for c in configs:
    summarize(c, all_results[c])

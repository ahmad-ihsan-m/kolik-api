"""
Fungsi inti untuk API prediksi kolik kuda.

Isinya SAMA dengan notebook PBL_PPKB_IPB_Halter_Asist.ipynb (A3, B2-B3, B6),
supaya fitur yang dihitung API identik dengan fitur saat model dilatih.
Kalau rumus fitur di notebook diubah, ubah juga di sini.
"""
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Bagian A: fitur pemeriksaan klinis (notebook A3)
# ----------------------------------------------------------------------------
NAMA_KLINIS = {
    "pulse": "detak jantung", "pulse_dev": "detak jantung di luar normal", "respiratory_rate": "laju napas",
    "respiratory_rate_dev": "laju napas di luar normal", "rectal_temp": "suhu tubuh",
    "rectal_temp_dev": "suhu tubuh di luar normal", "packed_cell_volume": "hematokrit (PCV)",
    "total_protein": "protein total", "mucous_membrane": "warna membran mukosa",
    "capillary_refill_time": "waktu pengisian kapiler", "temp_of_extremities": "suhu ekstremitas",
    "peripheral_pulse": "pulsa perifer", "pain": "tingkat nyeri", "peristalsis": "gerak usus",
    "abdominal_distention": "perut kembung", "abdomen": "temuan perut", "rectal_exam_feces": "feses (pemeriksaan rektal)",
    "nasogastric_tube": "gas lambung (selang NG)", "nasogastric_reflux": "refluks nasogastrik",
    "nasogastric_reflux_ph": "pH refluks", "abdomo_appearance": "cairan abdominosentesis",
    "abdomo_protein": "protein cairan perut", "age": "usia", "n_tanda_syok": "jumlah tanda syok",
}


LABEL_KATEGORI = {
    "age": {"adult": "Dewasa", "young": "Muda (< 6 bulan)"},
    "temp_of_extremities": {"normal": "Normal", "warm": "Hangat", "cool": "Agak dingin", "cold": "Dingin"},
    "peripheral_pulse": {"normal": "Normal", "increased": "Meningkat", "reduced": "Melemah", "absent": "Tidak teraba"},
    "mucous_membrane": {"normal_pink": "Merah muda normal", "bright_pink": "Merah muda terang", "pale_pink": "Merah muda pucat",
                        "pale_cyanotic": "Pucat kebiruan", "bright_red": "Merah terang", "dark_cyanotic": "Kebiruan gelap"},
    "capillary_refill_time": {"less_3_sec": "< 3 detik", "3": "3 detik", "more_3_sec": "> 3 detik"},
    "pain": {"alert": "Waspada, tidak nyeri", "depressed": "Lesu", "mild_pain": "Nyeri ringan hilang-timbul",
             "severe_pain": "Nyeri berat hilang-timbul", "extreme_pain": "Nyeri berat terus-menerus"},
    "peristalsis": {"hypermotile": "Hipermotil", "normal": "Normal", "hypomotile": "Berkurang", "absent": "Tidak ada"},
    "abdominal_distention": {"none": "Tidak ada", "slight": "Ringan", "moderate": "Sedang", "severe": "Berat"},
    "nasogastric_tube": {"none": "Tidak ada", "slight": "Sedikit", "significant": "Banyak"},
    "nasogastric_reflux": {"none": "Tidak ada", "less_1_liter": "< 1 liter", "more_1_liter": "> 1 liter"},
    "rectal_exam_feces": {"normal": "Normal", "increased": "Bertambah", "decreased": "Berkurang", "absent": "Tidak ada"},
    "abdomen": {"normal": "Normal", "other": "Lainnya", "firm": "Feses keras di usus besar",
                "distend_small": "Usus halus membesar", "distend_large": "Usus besar membesar"},
    "abdomo_appearance": {"clear": "Jernih", "cloudy": "Keruh", "serosanguious": "Bercampur darah"},
}


def deviation(x, lo, hi):
    # jarak keluar dari rentang normal; 0 jika di dalam rentang
    return np.where(x > hi, x - hi, np.where(x < lo, lo - x, 0.0))


def clinical_features(record, ordinal_maps, normal_ranges, feature_order):
    """record: dict mentah (nama kolom sama dengan horse.csv). Hasil: DataFrame 1 baris berisi fitur model."""
    X = pd.DataFrame([record])
    for col, mapping in ordinal_maps.items():
        if col in X.columns:
            X[col] = X[col].map(
                lambda v: mapping.get(str(v).strip().lower()) if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan
            ).astype(float)
        else:
            X[col] = np.nan
    for col, (lo, hi) in normal_ranges.items():
        X[col] = pd.to_numeric(X[col], errors="coerce") if col in X.columns else np.nan
        X[f"{col}_dev"] = pd.Series(deviation(X[col].astype(float).values, lo, hi), index=X.index).where(X[col].notna())
    X["n_tanda_syok"] = (X[["temp_of_extremities", "peripheral_pulse", "capillary_refill_time"]].fillna(0) >= 2).sum(axis=1)
    for c in feature_order:
        if c not in X.columns:
            X[c] = np.nan
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X[feature_order].astype(float)


def clinical_value_text(feat, value, ordinal_maps):
    if value is None or pd.isna(value):
        return "tidak diperiksa"
    if feat in ordinal_maps:
        key = {k: lab for lab, k in ordinal_maps[feat].items()}.get(int(value))
        return LABEL_KATEGORI.get(feat, {}).get(key, key if key is not None else f"{value:g}")
    return f"{value:g}"


def describe_clinical(feat, value, ordinal_maps):
    nama = NAMA_KLINIS.get(feat, feat)
    if value is None or pd.isna(value):
        return f"{nama} tidak diperiksa"
    sep = ": " if feat in ordinal_maps else " = "
    return f"{nama}{sep}{clinical_value_text(feat, value, ordinal_maps)}"


# ----------------------------------------------------------------------------
# Bagian B: fitur sensor per jam (notebook B2-B3)
# ----------------------------------------------------------------------------
SIG = ["hr", "rr", "temp", "spo2", "act", "eat", "lie"]
NORMAL_MID = dict(hr=36, rr=17, temp=37.75)
SENSOR_INPUT_COLS = ["hour", "hr", "rr", "temp", "spo2", "act", "eat", "lie", "barn_temp", "humidity", "nh3", "missing"]


# Kuantil tanda vital saat kolik dari data Kaggle (dipakai simulator demo)
PULSE_Q = [48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 52.0, 54.0, 60.0, 60.0, 64.0, 68.0,
           72.0, 80.0, 84.0, 88.0, 96.0, 100.0, 113.2, 125.2, 140.0]
RESP_Q = [16.0, 16.0, 16.0, 16.0, 16.0, 18.0, 20.0, 20.0, 24.0, 24.0, 25.0, 28.0,
          30.0, 32.0, 35.0, 36.0, 40.0, 42.0, 51.0, 70.0, 70.0]


def sample_from_quantiles(q, rng, n=1):
    return np.interp(rng.random(n), np.linspace(0, 1, len(q)), q)


def simulate_horse(hid, cfg, rng, forced=None, cal_pulse=PULSE_Q, cal_resp=RESP_Q):
    """Simulator notebook B1. `forced` = daftar episode kolik yang ditentukan sendiri (mode demo)."""
    spd = 24 * 60 // cfg["step_min"]
    T = cfg["days"] * spd
    th = np.arange(T) * cfg["step_min"] / 60
    hod = th % 24
    day = (th // 24).astype(int)

    b = dict(hr=rng.uniform(26, 46), rr=rng.uniform(9, 20), temp=rng.uniform(37.3, 38.2),
             spo2=rng.uniform(95.5, 98.5), act=rng.uniform(0.6, 1.4), eat=rng.uniform(6, 14), lie=rng.uniform(0.2, 0.6))
    drift = np.repeat(np.cumsum(rng.normal(0, 0.6, cfg["days"])), spd)
    drift = drift - drift.mean()
    circ = np.sin(2 * np.pi * (hod - 9) / 24)
    daytime = ((hod >= 6) & (hod <= 19)).astype(float)

    day_off = rng.normal(0, 1.2, cfg["days"]) + 4.5 * (rng.random(cfg["days"]) < cfg["heat_day_prob"])
    barn_t = 27 + 3.5 * circ + day_off[day] + rng.normal(0, 0.4, T)
    hum = np.clip(82 - 2.2 * (barn_t - 27) + rng.normal(0, 2, T), 45, 98)
    nh3 = np.clip(rng.uniform(4, 12) + 6 * np.sin(2 * np.pi * day / 7) ** 2 + rng.normal(0, 1, T), 0, None)
    heat = np.clip(barn_t - 29, 0, None)

    hr = b["hr"] + drift + 2.5 * circ + 0.35 * heat + rng.normal(0, 3.0, T)
    rr = b["rr"] + 0.3 * drift + 1.0 * circ + 0.9 * heat + rng.normal(0, 2.0, T)
    temp = b["temp"] + 0.18 * circ + 0.05 * heat + rng.normal(0, 0.10, T)
    spo2 = b["spo2"] + rng.normal(0, 0.5, T)
    act = b["act"] * (0.35 + 0.45 * daytime) + rng.normal(0, 0.06, T)
    feed = sum(np.exp(-0.5 * ((hod - h) / 0.6) ** 2) for h in (7, 12, 17, 21))
    eat = b["eat"] * feed * rng.lognormal(0, 0.15, cfg["days"])[day] + 1.2 * daytime + rng.normal(0, 1.0, T)
    lie = b["lie"] * (1 - daytime) + 0.08 * daytime + rng.normal(0, 0.06, T)

    for _ in range(rng.poisson(cfg["excitement_per_day"] * cfg["days"])):
        s = rng.integers(0, T); L = int(rng.uniform(0.5, 1.5) * 60 / cfg["step_min"])
        hr[s:s + L] += rng.uniform(15, 30); rr[s:s + L] += rng.uniform(5, 12); act[s:s + L] += rng.uniform(0.5, 1.0)

    for _ in range(rng.poisson(cfg["malaise_per_day"] * cfg["days"])):
        s = rng.integers(0, T); L = int(rng.uniform(4, 12) * 60 / cfg["step_min"])
        k = np.sin(np.linspace(0, np.pi, len(eat[s:s + L])))
        eat[s:s + L] *= 1 - 0.5 * k; lie[s:s + L] += 0.2 * k; hr[s:s + L] += rng.uniform(2, 6) * k; act[s:s + L] -= 0.15 * k

    phase = np.zeros(T, dtype=np.int8)
    episodes = []
    plan = forced or []
    for ev in plan:
        o = ev["onset_h"]
        P = ev.get("prodrome_h") or rng.uniform(*cfg["prodrome_h"])
        A = ev.get("active_h") or rng.uniform(*cfg["active_h"])
        R = cfg["recovery_h"]
        sev = ev.get("severity") or rng.uniform(0.6, 1.2)
        tgt_hr = float(sample_from_quantiles(cal_pulse, rng)[0])
        tgt_rr = float(sample_from_quantiles(cal_resp, rng)[0])
        pro = (th >= o - P) & (th < o)
        p = np.zeros(T); p[pro] = ((th[pro] - (o - P)) / P) ** 1.5
        actv = (th >= o) & (th < o + A)
        rec = (th >= o + A) & (th < o + A + R)
        p[actv] = 1.0
        p[rec] = 1 - (th[rec] - (o + A)) / R
        hr += p * sev * rng.uniform(4, 10) + actv * (tgt_hr - b["hr"]) * 0.9 + rec * p * (tgt_hr - b["hr"]) * 0.4
        rr += p * sev * 3 + actv * (tgt_rr - b["rr"]) * 0.8
        temp += p * sev * 0.2
        spo2 -= p * sev * 0.8
        eat *= (1 - 0.6 * p * min(sev, 1))
        lie += 0.22 * p * sev
        act += -0.18 * p * (~actv) + actv * rng.uniform(0.3, 0.9, T) * (rng.random(T) < 0.4)
        phase[pro] = np.maximum(phase[pro], 1); phase[actv] = 2; phase[rec] = np.where(phase[rec] == 0, 3, phase[rec])
        episodes.append(dict(onset_h=o, prodrome_h=P, active_h=A, severity=sev))

    df = pd.DataFrame(dict(horse=hid, t_h=th, hr=hr, rr=rr, temp=temp, spo2=np.clip(spo2, 85, 100),
                           act=np.clip(act, 0, None), eat=np.clip(eat, 0, 15), lie=np.clip(lie, 0, 1),
                           barn_temp=barn_t, humidity=hum, nh3=nh3, phase=phase))
    for _ in range(rng.poisson(cfg["dropout_per_day"] * cfg["days"])):
        s = rng.integers(0, T); L = int(rng.uniform(1, 6) * 60 / cfg["step_min"])
        df.loc[s:s + L, SIG] = np.nan
    return df, episodes


def to_hourly(raw):
    """Agregasi 15 menit -> per jam (notebook B2)."""
    raw = raw.copy()
    raw["hour"] = (raw.t_h // 1).astype(int)
    agg = dict(hr="mean", rr="mean", temp="mean", spo2="mean", act="mean", eat="sum", lie="mean",
               barn_temp="mean", humidity="mean", nh3="mean", phase="max")
    hourly = raw.groupby(["horse", "hour"]).agg(agg).reset_index()
    hourly["missing"] = raw.groupby(["horse", "hour"]).hr.apply(lambda s: s.isna().mean()).values
    hourly.loc[hourly.missing > 0.75, "eat"] = np.nan
    hourly["hod"] = hourly.hour % 24
    return hourly


def prepare_hourly(records, horse=0):
    """records: list dict per jam (lihat SENSOR_INPUT_COLS). Jam yang bolong diisi baris kosong."""
    df = pd.DataFrame(records)
    for c in SENSOR_INPUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[SENSOR_INPUT_COLS].copy()
    df["hour"] = df["hour"].astype(int)
    df = df.drop_duplicates("hour", keep="last").set_index("hour").sort_index()
    df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1))   # jam hilang -> NaN
    df.index.name = "hour"
    df = df.reset_index()
    df["missing"] = df["missing"].fillna(df[SIG].isna().all(axis=1).astype(float))
    df["horse"] = horse
    df["hod"] = df.hour % 24
    return df


def add_features(hourly):
    df = hourly.sort_values(["horse", "hour"]).reset_index(drop=True)
    g = df.groupby("horse")
    for s in SIG:
        df[f"{s}_3h"] = g[s].transform(lambda x: x.rolling(3, min_periods=2).mean())
        df[f"{s}_6h"] = g[s].transform(lambda x: x.rolling(6, min_periods=3).mean())
        df[f"{s}_slope6"] = df[f"{s}_3h"] - g[f"{s}_3h"].shift(3)
    for s, mid in NORMAL_MID.items():
        df[f"{s}_popdev"] = df[f"{s}_3h"] - mid
    df["missing_6h"] = g["missing"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df["hod_sin"], df["hod_cos"] = np.sin(2 * np.pi * df.hod / 24), np.cos(2 * np.pi * df.hod / 24)
    gh = df.groupby(["horse", "hod"])
    for s in SIG:
        df[f"{s}_base"] = gh[s].transform(lambda x: x.shift(1).rolling(7, min_periods=3).median())
        resid = df[s] - df[f"{s}_base"]
        sd = resid.groupby(df.horse).transform(lambda x: x.shift(1).rolling(168, min_periods=48).std())
        df[f"{s}_z"] = (df[f"{s}_3h"] - df[f"{s}_base"]) / (sd + 1e-6)
        df[f"{s}_z6"] = df.groupby("horse")[f"{s}_z"].transform(lambda x: x.rolling(6, min_periods=3).mean())
    return df


def alarm_flags(prob, thr, consec=2):
    """Alarm = probabilitas >= ambang selama `consec` jam berturut-turut."""
    p = pd.Series(np.asarray(prob) >= thr)
    return (p.rolling(consec, min_periods=consec).sum() >= consec).values


# ----------------------------------------------------------------------------
# Kalimat alasan alarm sensor (notebook B6)
# ----------------------------------------------------------------------------
LABEL = dict(hr="detak jantung", rr="laju napas", temp="suhu tubuh", spo2="SpO2", act="aktivitas",
             eat="waktu makan", lie="waktu berbaring")
UNIT = dict(hr="bpm", rr="/menit", temp="°C", spo2="%", act="", eat="menit/jam", lie="")


def describe_sensor(feat, row):
    s, _, kind = feat.partition("_")
    if s not in LABEL:
        return {"barn": f"suhu kandang {row.get('barn_temp', np.nan):.1f} °C",
                "humidity": f"kelembapan {row.get('humidity', np.nan):.0f}%",
                "nh3": f"NH3 kandang {row.get('nh3', np.nan):.1f} ppm",
                "missing": "data sensor banyak hilang", "hod": "jam dalam sehari"}.get(s, feat)
    if kind in ("z", "z6"):
        z = row[feat]
        arah = "naik" if z > 0 else "turun"
        zt = f"{abs(z):.1f}" if abs(z) <= 5 else "lebih dari 5"
        now, base = row.get(f"{s}_3h"), row.get(f"{s}_base")
        extra = (f" ({now:.1f} vs normalnya {base:.1f} {UNIT[s]})"
                 if pd.notna(base) and pd.notna(now) and s in ("hr", "rr", "temp", "eat") else "")
        return f"{LABEL[s]} {arah} {zt} SD dari normal kuda ini{extra}"
    if kind == "slope6":
        return f"{LABEL[s]} {'meningkat' if row[feat] > 0 else 'menurun'} dalam 6 jam terakhir"
    return f"{LABEL[s]} {row[feat]:.1f} {UNIT[s]}".strip()


def top_sensor_reasons(contrib_row, feats, row, top=3):
    order = np.argsort(-np.asarray(contrib_row))
    reasons, seen = [], set()
    for j in order:
        if contrib_row[j] <= 0 or len(reasons) == top:
            break
        f = feats[j]
        base = f.split("_")[0]
        if base in seen:
            continue
        seen.add(base)
        reasons.append(describe_sensor(f, row))
    return reasons

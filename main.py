"""
API Prediksi Kolik Kuda — PBL MK PPKB

Model dilatih di notebook PBL_PPKB_IPB_Halter_Asist.ipynb (bagian B8 mengekspor artifacts.zip).

Lokal :  pip install -r requirements.txt uvicorn  &&  uvicorn main:app --reload
Docs  :  http://localhost:8000/docs
Deploy:  push ke GitHub lalu import di Vercel (Vercel otomatis mengenali `app` di main.py)
"""
import gzip
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import kolik_core as kc

ART = Path(__file__).parent / "artifacts"

# ----------------------------------------------------------------------------
# Muat model dan data sekali saat fungsi dinyalakan (cold start)
# ----------------------------------------------------------------------------
KLINIS = joblib.load(ART / "model_risiko_kolik.joblib")
SENSOR = joblib.load(ART / "model_prediksi_kolik_6jam.joblib")
THR = float(SENSOR["threshold"])
CFG = SENSOR["cfg"]
with gzip.open(ART / "kandang.json.gz", "rt") as _f:
    KANDANG = json.load(_f)
HORSES = {h["id"]: h for h in KANDANG["horses"]}
TEST_H = set(KANDANG.get("test_horses", []))
BURN_H = KANDANG["burn_in_days"] * 24
N_DAYS = KANDANG["days"]

app = FastAPI(
    title="API Prediksi Kolik Kuda",
    description="Model A: risiko fatal dari pemeriksaan klinis (data Kaggle). "
                "Model B: risiko kolik 6 jam ke depan dari data sensor per jam (simulasi).",
    version="2.0",
)

# Izinkan website memanggil API. Isi ALLOWED_ORIGINS di Vercel, mis.
# "https://kolik-web.vercel.app,http://localhost:3000". Kosong = semua origin boleh.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins or ["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------------
def _num(v, nd=3):
    """Angka -> float bulat; NaN/inf -> None (aman untuk JSON)."""
    if v is None:
        return None
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


def _col(s, nd=3):
    return [_num(v, nd) for v in s]


def fmt_time(hour):
    return f"Hari {int(hour // 24)}, {int(hour % 24):02d}:00"


def _xgb_contribs(model, X):
    """SHAP TreeExplainer versi bawaan XGBoost (hasil sama, tanpa library shap)."""
    import xgboost as xgb
    return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]


def _is_xgb(m):
    return type(m).__name__.startswith("XGB")


def status_of(risk, alarm, phase):
    if phase == 2:
        return "KOLIK AKTIF"
    if alarm:
        return "ALARM"
    if risk is None or pd.isna(risk):
        return "MEMBANGUN BASELINE"
    if risk >= THR / 2:
        return "WASPADA"
    return "NORMAL"


def _score(df):
    """Tambah kolom risk (NaN selama masa membangun baseline) dan alarm."""
    df["risk"] = SENSOR["model"].predict_proba(df[SENSOR["features"]].astype(float))[:, 1]
    df.loc[df.hour < BURN_H, "risk"] = np.nan
    df["alarm"] = kc.alarm_flags(df["risk"].fillna(0).values, THR)
    return df


def _reasons(df, idx, top=3):
    feats = SENSOR["features"]
    c = _xgb_contribs(SENSOR["model"], df.loc[idx, feats].astype(float))
    return [kc.top_sensor_reasons(c[k], feats, df.loc[i].to_dict(), top) for k, i in enumerate(idx)]


def _alarm_starts(df):
    a = df["alarm"].values
    return [df.index[k] for k in range(len(df)) if a[k] and (k == 0 or not a[k - 1])]


@lru_cache(maxsize=40)
def horse_frame(hid):
    """Semua fitur + skor untuk satu kuda kandang (dihitung sekali lalu disimpan di memori)."""
    h = HORSES[hid]
    df = pd.DataFrame(h["cols"]).astype({"hour": int})
    df["horse"] = hid
    df["hod"] = df.hour % 24
    df = kc.add_features(df.drop(columns=["hr_3h", "hr_base", "risk"]))
    return _score(df)


# ----------------------------------------------------------------------------
# Info umum
# ----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    """Informasi untuk membangun tampilan website."""
    m = KLINIS["model"]
    return {
        "klinis": {
            "model": type(m[-1] if hasattr(m, "steps") else m).__name__,
            "threshold": KLINIS["threshold"],
            "pilihan_kategori": {k: list(v.keys()) for k, v in KLINIS["ordinal_maps"].items()},
            "label_kategori": kc.LABEL_KATEGORI,
            "rentang_normal": {k: list(v) for k, v in KLINIS["normal_ranges"].items()},
            "nama_indonesia": kc.NAMA_KLINIS,
        },
        "sensor": {"threshold": THR, "horizon_jam": SENSOR["horizon_h"], "hari": N_DAYS,
                   "hari_baseline": KANDANG["burn_in_days"], "jumlah_kuda": len(HORSES),
                   "kolom_input": kc.SENSOR_INPUT_COLS},
    }


# ----------------------------------------------------------------------------
# Model A: pemeriksaan klinis
# ----------------------------------------------------------------------------
class PemeriksaanKlinis(BaseModel):
    """Semua field boleh kosong (null). Nilai kategori harus salah satu dari /meta -> pilihan_kategori."""
    age: Optional[Literal["adult", "young"]] = None
    rectal_temp: Optional[float] = Field(None, description="°C")
    pulse: Optional[float] = Field(None, description="detak/menit")
    respiratory_rate: Optional[float] = Field(None, description="napas/menit")
    temp_of_extremities: Optional[str] = None
    peripheral_pulse: Optional[str] = None
    mucous_membrane: Optional[str] = None
    capillary_refill_time: Optional[str] = None
    pain: Optional[str] = None
    peristalsis: Optional[str] = None
    abdominal_distention: Optional[str] = None
    nasogastric_tube: Optional[str] = None
    nasogastric_reflux: Optional[str] = None
    nasogastric_reflux_ph: Optional[float] = None
    rectal_exam_feces: Optional[str] = None
    abdomen: Optional[str] = None
    packed_cell_volume: Optional[float] = Field(None, description="%")
    total_protein: Optional[float] = Field(None, description="g/dL")
    abdomo_appearance: Optional[str] = None
    abdomo_protein: Optional[float] = None

    model_config = {"json_schema_extra": {"example": {
        "age": "adult", "rectal_temp": 37.2, "pulse": 104, "respiratory_rate": 36,
        "temp_of_extremities": "cold", "peripheral_pulse": "reduced", "mucous_membrane": "dark_cyanotic",
        "capillary_refill_time": "more_3_sec", "pain": "extreme_pain", "peristalsis": "absent",
        "abdominal_distention": "severe", "nasogastric_reflux": "more_1_liter", "abdomen": "distend_small",
        "packed_cell_volume": 62, "total_protein": 7.5}}}


@app.post("/klinis/predict")
def klinis_predict(data: PemeriksaanKlinis, top: int = 3):
    record = data.model_dump()
    salah = {k: v for k, v in record.items()
             if v is not None and k in KLINIS["ordinal_maps"] and str(v).strip().lower() not in KLINIS["ordinal_maps"][k]}
    if salah:
        raise HTTPException(422, {"pesan": "Nilai kategori tidak dikenal", "field": salah,
                                  "pilihan": {k: list(KLINIS["ordinal_maps"][k]) for k in salah}})

    feats, om = KLINIS["features"], KLINIS["ordinal_maps"]
    X = kc.clinical_features(record, om, KLINIS["normal_ranges"], feats)
    model, thr = KLINIS["model"], KLINIS["threshold"]
    prob = float(model.predict_proba(X)[:, 1][0])

    # Kontribusi fitur: XGBoost -> SHAP bawaan; model lain -> uji ganti nilai dengan median data latih
    if _is_xgb(model):
        contrib = _xgb_contribs(model, X)[0]
        metode = "SHAP (satuan log-odds)"
    else:
        imp = getattr(model, "named_steps", {}).get("imp")
        med = imp.statistics_ if imp is not None else np.full(len(feats), np.nan)
        Xp = pd.DataFrame(np.repeat(X.values, len(feats), axis=0), columns=feats)
        for j in range(len(feats)):
            Xp.iat[j, j] = med[j]
        contrib = prob - model.predict_proba(Xp)[:, 1]
        metode = "perkiraan: perubahan peluang jika nilai diganti median data latih"

    alasan = [{"fitur": feats[j], "teks": kc.describe_clinical(feats[j], X.iloc[0, j], om),
               "kontribusi": round(float(contrib[j]), 4)}
              for j in np.argsort(-contrib)[:top] if contrib[j] > 0]
    faktor = [{"fitur": feats[j], "nama": kc.NAMA_KLINIS.get(feats[j], feats[j]),
               "nilai": kc.clinical_value_text(feats[j], X.iloc[0, j], om),
               "arah": "naik" if contrib[j] > 0 else "turun", "besar": round(float(contrib[j]), 4)}
              for j in np.argsort(-np.abs(contrib))[:6] if contrib[j] != 0]

    n_isi = sum(v is not None for v in record.values())
    status = "BERISIKO TINGGI" if prob >= thr else "risiko lebih rendah"
    kalimat = f"{status} (peluang fatal {prob:.0%})."
    if alasan:
        kalimat += " Penyumbang utama: " + "; ".join(a["teks"] for a in alasan) + "."
    return {"probabilitas_fatal": round(prob, 4), "threshold": thr, "berisiko_tinggi": prob >= thr,
            "status": status, "kalimat": kalimat, "alasan": alasan, "faktor": faktor,
            "metode_penjelasan": metode, "jumlah_field_terisi": n_isi, "jumlah_field": len(record),
            "catatan": None if n_isi >= 5 else "Isi minimal 5 pemeriksaan (sebaiknya termasuk detak jantung) supaya perkiraan bermakna."}


# ----------------------------------------------------------------------------
# Model B: halaman Ringkasan kandang
# ----------------------------------------------------------------------------
@app.get("/kandang")
def kandang(hari: int = 27, jam: int = 12):
    """Status semua kuda pada satu titik waktu + alarm 24 jam terakhir."""
    hari = int(np.clip(hari, KANDANG["burn_in_days"], N_DAYS - 1))
    t = hari * 24 + int(np.clip(jam, 0, 23))
    cards, alarms = [], []
    for hid, h in HORSES.items():
        c = h["cols"]
        k0 = c["hour"].index(t) if t in c["hour"] else None
        if k0 is None:
            continue
        risk = pd.Series(c["risk"], dtype=float)
        al = kc.alarm_flags(risk.fillna(0).values, THR)
        r = c["risk"][k0]
        cards.append({"id": hid, "uji": hid in TEST_H, "status": status_of(r, al[k0], c["phase"][k0]),
                      "risiko": r, "hr_3h": c["hr_3h"][k0], "hr_base": c["hr_base"][k0]})
        win = [k for k in range(max(0, k0 - 23), k0 + 1) if al[k]]
        starts = [k for k in win if k - 1 not in win]
        if starts:
            df = horse_frame(hid)
            idx = [df.index[df.hour == c["hour"][k]][0] for k in starts]
            for i, why in zip(idx, _reasons(df, idx)):
                alarms.append({"hour": int(df.hour[i]), "waktu": fmt_time(df.hour[i]), "kuda": hid,
                               "risiko": _num(df.risk[i]), "alasan": why})
    order = {"KOLIK AKTIF": 0, "ALARM": 1, "WASPADA": 2, "NORMAL": 3, "MEMBANGUN BASELINE": 4}
    cards.sort(key=lambda x: (order[x["status"]], -(x["risiko"] or 0)))
    hitung = {s: sum(c["status"] == s for c in cards) for s in order}
    return {"t": t, "waktu": fmt_time(t), "threshold": THR, "jumlah": hitung, "kuda": cards,
            "alarm_24jam": sorted(alarms, key=lambda a: -a["hour"])}


# ----------------------------------------------------------------------------
# Model B: halaman Detail kuda
# ----------------------------------------------------------------------------
@app.get("/kuda")
def daftar_kuda():
    return [{"id": hid, "uji": hid in TEST_H, "episode": h["episodes"]} for hid, h in sorted(HORSES.items())]


def _series(f):
    return {"hour": [int(v) for v in f.hour], "hr_3h": _col(f.hr_3h, 1), "hr_base": _col(f.hr_base, 1),
            "lie_3h": _col(f.lie_3h), "lie_base": _col(f.lie_base), "eat": _col(f.eat, 2),
            "eat_base": _col(f.eat_base, 2), "risk": _col(f.risk, 4), "alarm": [bool(v) for v in f.alarm]}


@app.get("/kuda/{hid}")
def detail_kuda(hid: int, sampai_hari: Optional[int] = None, rentang: int = 7):
    if hid not in HORSES:
        raise HTTPException(404, f"Kuda {hid} tidak ada")
    eps = HORSES[hid]["episodes"]
    if sampai_hari is None:
        sampai_hari = int(min(N_DAYS, eps[0]["onset_h"] // 24 + 2)) if eps else KANDANG["burn_in_days"] + 14
    sampai_hari = int(np.clip(sampai_hari, 1, N_DAYS))
    rentang = int(np.clip(rentang, 1, 30))
    df = horse_frame(hid)
    f = df[(df.hour >= (sampai_hari - rentang) * 24) & (df.hour < sampai_hari * 24)]
    starts = _alarm_starts(f)
    out = []
    for i, why in zip(starts, _reasons(df, starts) if starts else []):
        h = int(df.hour[i])
        nxt = [e for e in eps if 0 <= e["onset_h"] - h <= 24]
        during = [e for e in eps if e["onset_h"] <= h <= e["onset_h"] + e["active_h"] + 36]
        if nxt:
            nilai, teks = "benar", f"Benar: kolik {nxt[0]['onset_h'] - (h + 1):.1f} jam kemudian"
        elif during:
            nilai, teks = "saat", "Saat/sesudah kolik"
        else:
            nilai, teks = "palsu", "Alarm palsu"
        out.append({"hour": h, "waktu": fmt_time(h), "risiko": _num(df.risk[i]), "alasan": why,
                    "penilaian": nilai, "penilaian_teks": teks})
    return {"id": hid, "uji": hid in TEST_H, "threshold": THR, "sampai_hari": sampai_hari, "rentang": rentang,
            "episode": eps, "series": _series(f), "alarm": out}


# ----------------------------------------------------------------------------
# Model B: halaman Demo suntik kolik
# ----------------------------------------------------------------------------
class Suntik(BaseModel):
    prodromal_jam: float = Field(10, ge=3, le=16, description="lama fase prodromal (jam)")
    keparahan: float = Field(0.9, ge=0.6, le=1.2)
    seed: int = Field(7, ge=0, le=9999)


@app.post("/demo/suntik")
def demo_suntik(p: Suntik):
    """Simulasikan kuda baru 12 hari, suntik satu episode kolik, hitung skor tiap jam.
    Website memutar hasilnya jam demi jam."""
    rng = np.random.default_rng(p.seed)
    cfg = dict(CFG, days=12)
    onset = 10 * 24 + float(rng.uniform(0, 12))
    raw, ep = kc.simulate_horse(999, cfg, rng, forced=[dict(onset_h=onset, prodrome_h=p.prodromal_jam, severity=p.keparahan)])
    df = _score(kc.add_features(kc.to_hourly(raw)))
    start, stop = int(onset) - 30, int(onset) + 8
    f = df[(df.hour >= start - 48) & (df.hour <= stop)]
    pre = f[f.alarm & (f.hour + 1 <= onset) & (f.hour + 1 > onset - 24)]
    first = None
    if len(pre):
        i = pre.index[0]
        first = {"hour": int(df.hour[i]), "waktu": fmt_time(df.hour[i]), "risiko": _num(df.risk[i]),
                 "lead_jam": round(onset - (df.hour[i] + 1), 1), "alasan": _reasons(df, [i])[0]}
    starts = _alarm_starts(f)
    return {"onset_h": round(onset, 2), "onset_waktu": fmt_time(onset), "mulai_putar": start, "selesai_putar": stop,
            "threshold": THR, "episode": [{k: round(float(v), 2) for k, v in e.items()} for e in ep],
            "series": _series(f), "alarm_pertama": first,
            "alasan_per_alarm": {int(df.hour[i]): w for i, w in zip(starts, _reasons(df, starts) if starts else [])}}


# ----------------------------------------------------------------------------
# Model B: data sensor sungguhan (untuk integrasi alat nanti)
# ----------------------------------------------------------------------------
class DataJam(BaseModel):
    hour: int = Field(..., description="jam ke- (bilangan bulat berurutan, mis. jam sejak pemasangan alat)")
    hr: Optional[float] = Field(None, description="detak jantung rata-rata jam ini (bpm)")
    rr: Optional[float] = Field(None, description="laju napas (/menit)")
    temp: Optional[float] = Field(None, description="suhu tubuh (°C)")
    spo2: Optional[float] = None
    act: Optional[float] = Field(None, description="indeks aktivitas")
    eat: Optional[float] = Field(None, description="menit makan dalam jam ini")
    lie: Optional[float] = Field(None, description="proporsi waktu berbaring 0-1")
    barn_temp: Optional[float] = None
    humidity: Optional[float] = None
    nh3: Optional[float] = None
    missing: Optional[float] = Field(None, description="proporsi data sensor hilang 0-1")


class InputSensor(BaseModel):
    records: List[DataJam] = Field(..., min_length=24, max_length=24 * 120)


@app.post("/sensor/predict")
def sensor_predict(data: InputSensor, top: int = 3):
    """Kirim data per jam satu kuda (disarankan ≥ 8 hari terakhir). Hasil: skor risiko tiap jam, alarm, dan alasannya."""
    df = kc.add_features(kc.prepare_hourly([r.model_dump() for r in data.records]))
    feats = SENSOR["features"]
    df["risk"] = SENSOR["model"].predict_proba(df[feats].astype(float))[:, 1]
    df["alarm"] = kc.alarm_flags(df["risk"].values, THR)
    starts = _alarm_starts(df)
    last = df.index[-1]
    n = len(df)
    return {
        "threshold": THR, "horizon_jam": SENSOR["horizon_h"],
        "catatan": None if n >= 24 * 8 else f"Data baru {n} jam. Baseline individu butuh sekitar 8 hari, jadi skor awal kurang andal.",
        "terkini": {"hour": int(df.hour[last]), "risiko": _num(df.risk[last], 4), "alarm": bool(df.alarm[last]),
                    "alasan": _reasons(df, [last], top)[0]},
        "alarm": [{"hour": int(df.hour[i]), "risiko": _num(df.risk[i], 4), "alasan": w}
                  for i, w in zip(starts, _reasons(df, starts, top) if starts else [])],
        "timeline": [{"hour": int(h), "risiko": _num(p, 4), "alarm": bool(a)} for h, p, a in zip(df.hour, df.risk, df.alarm)],
    }

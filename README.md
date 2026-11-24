# kolik-api — API Prediksi Kolik Kuda

API FastAPI yang membungkus dua model dari notebook `PBL_PPKB_IPB_Halter_Asist.ipynb`:

| Endpoint | Fungsi | Halaman website |
|---|---|---|
| `GET /meta` | Pilihan kategori + label Indonesia, rentang normal, ambang | semua |
| `GET /kandang?hari=&jam=` | Status 30 kuda pada satu waktu + alarm 24 jam terakhir dan alasannya | Ringkasan kandang |
| `GET /kuda` | Daftar kuda + episode kolik | Detail kuda |
| `GET /kuda/{id}?sampai_hari=&rentang=` | Grafik HR/berbaring/makan vs baseline, skor risiko, alarm + penilaian benar/palsu | Detail kuda |
| `POST /demo/suntik` | Simulasi kuda baru + kolik yang disuntikkan, skor tiap jam | Demo suntik kolik |
| `POST /klinis/predict` | Model A: peluang fatal + faktor berpengaruh | Pemeriksaan klinis |
| `POST /sensor/predict` | Model B untuk data sensor sungguhan (integrasi alat nanti) | – |
| `GET /docs` | Dokumentasi interaktif (bisa dicoba langsung) | – |

## 1. Isi model dari Colab

1. Di Colab, jalankan notebook sampai habis (**Runtime → Run all**). Cell terakhir mengunduh `artifacts.zip`.
2. Ekstrak `artifacts.zip`:
   - `model_risiko_kolik.joblib`, `model_prediksi_kolik_6jam.joblib`, `kandang.json.gz` → folder `artifacts/`
   - `requirements.txt` dan `.python-version` → folder utama ini (timpa yang lama)

Versi di `requirements.txt` sengaja disamakan dengan Colab. File `.joblib` bisa gagal dibuka atau salah prediksi
kalau versi scikit-learn/xgboost berbeda. Setiap kali model dilatih ulang, ulangi langkah ini.

## 2. Coba di komputer sendiri

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt uvicorn
uvicorn main:app --reload
```

Buka http://localhost:8000/docs dan coba endpoint `/klinis/predict` (sudah ada contoh isian).

## 3. Deploy ke Vercel

1. Push folder ini ke repo GitHub sendiri (mis. `kolik-api`).
2. Di vercel.com → **Add New → Project** → pilih repo itu → **Deploy**. Vercel otomatis mengenali FastAPI di `main.py`.
3. Setelah jadi, catat URL-nya, mis. `https://kolik-api.vercel.app`, lalu cek `https://kolik-api.vercel.app/docs`.
4. (Opsional) **Settings → Environment Variables**: `ALLOWED_ORIGINS=https://kolik-web.vercel.app`
   supaya hanya website kalian yang boleh memanggil API langsung dari browser.

Ukuran dependensi sekitar 280 MB (batas Vercel untuk Python 500 MB). Jangan ganti `xgboost-cpu` dengan
`xgboost`: versi biasa ikut membawa library GPU dan melebihi batas.

Panggilan pertama setelah lama tidak dipakai bisa lambat beberapa detik (cold start: memuat pandas, sklearn, model).

## Contoh panggilan

```bash
curl -X POST https://kolik-api.vercel.app/klinis/predict \
  -H "Content-Type: application/json" \
  -d '{"pulse": 88, "respiratory_rate": 36, "rectal_temp": 38.6, "pain": "severe_pain",
       "temp_of_extremities": "cool", "peripheral_pulse": "reduced", "packed_cell_volume": 55}'
```

Data sensor untuk `/sensor/predict`: daftar per jam satu kuda, minimal 24 jam, disarankan ≥ 8 hari terakhir
(baseline individu dihitung dari 7 hari sebelumnya pada jam yang sama):

```json
{"records": [{"hour": 0, "hr": 36.2, "rr": 15.1, "temp": 37.6, "spo2": 97.1, "act": 0.4,
              "eat": 9.5, "lie": 0.3, "barn_temp": 27.4, "humidity": 80, "nh3": 8.1, "missing": 0}, ...]}
```

## Kalau rumus fitur di notebook diubah

`kolik_core.py` berisi salinan rumus fitur dari notebook (A3, B2–B3). Kalau notebook diubah, ubah juga file ini,
supaya fitur saat prediksi sama dengan fitur saat model dilatih.

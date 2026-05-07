"""
Gunluk Altin Fiyat Takip & Tahmin Sistemi
==========================================
Kurulum: pip install yfinance prophet pandas
Calistirma: python altin_gunluk.py

Her gun calistirinca:
  - Bugunun altin fiyatini Yahoo Finance'den ceker
  - altin_gunluk_veri.csv dosyasina kaydeder
  - Ertesi gun icin tahmin uretir
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
import pandas as pd
import numpy as np

VERI_DOSYASI    = "altin_gunluk_veri.csv"  # Fiyatlar buraya birikir
GECMIS_DOSYASI  = "altin_fiyat.csv"        # Elinizdeki 10 yillik veri
SEMBOL          = "GC=F"                   # Yahoo Finance altin vadeli islem kodu


# -- 0. Gecmis veriyi aktar (sadece ilk calistirmada) -------------------------
def gecmis_veriyi_aktar():
    import os
    if os.path.exists(VERI_DOSYASI):
        return  # Zaten olusturulmus, atliyoruz

    if not os.path.exists(GECMIS_DOSYASI):
        print(f"   Gecmis veri dosyasi bulunamadi ({GECMIS_DOSYASI}), atlanıyor.")
        return

    print(f"Gecmis veri aktariliyor: {GECMIS_DOSYASI} -> {VERI_DOSYASI}")

    with open(GECMIS_DOSYASI, "r", encoding="utf-8", errors="ignore") as f:
        ilk_satir = f.readline()
    ayrac = ";" if ilk_satir.count(";") > ilk_satir.count(",") else ","

    df = pd.read_csv(GECMIS_DOSYASI, sep=ayrac)
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]

    tarih_adaylari = ["date", "Date", "tarih", "Tarih", "DATE"]
    tarih_sutun = next((a for a in tarih_adaylari if a in df.columns), df.columns[0])

    fiyat_adaylari = ["close", "Close", "price", "Price", "Value", "value",
                      "fiyat", "Fiyat", "kapanis", "Kapanis"]
    fiyat_sutun = next((a for a in fiyat_adaylari if a in df.columns), None)
    if fiyat_sutun is None:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                fiyat_sutun = col
                break

    df = df[[tarih_sutun, fiyat_sutun]].copy()
    df.columns = ["date", "close"]
    df["date"]  = pd.to_datetime(df["date"], dayfirst=False).dt.date.astype(str)
    df["close"] = pd.to_numeric(df["close"].astype(str).str.replace(",", "."), errors="coerce")
    df.dropna(inplace=True)
    df.sort_values("date", inplace=True)
    df.drop_duplicates(subset="date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(VERI_DOSYASI, index=False)
    print(f"   {len(df)} gunluk gecmis veri aktarildi.")


# ── 1. Bugunun fiyatini cek ───────────────────────────────────────────────────
def bugunun_fiyatini_cek():
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance kurulu degil: pip install yfinance")
        sys.exit(1)

    print("Yahoo Finance'den altin fiyati cekiliyor...")
    ticker = yf.Ticker(SEMBOL)

    # Son 5 gunluk veri al (hafta sonu / tatil korumasi)
    df = ticker.history(period="5d")
    if df.empty:
        print("Veri cekemedim. Internet baglantinizi kontrol edin.")
        sys.exit(1)

    son_gun   = df.index[-1].date()
    son_fiyat = round(df["Close"].iloc[-1], 2)
    print(f"   Tarih : {son_gun}")
    print(f"   Fiyat : {son_fiyat} USD")
    return son_gun, son_fiyat


# ── 2. CSV'ye kaydet (mevcut varsa guncelle) ──────────────────────────────────
def kaydet(tarih, fiyat):
    yeni_satir = pd.DataFrame([{"date": str(tarih), "close": fiyat}])

    try:
        df = pd.read_csv(VERI_DOSYASI)
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

        if str(tarih) in df["date"].values:
            df.loc[df["date"] == str(tarih), "close"] = fiyat
            print(f"   Guncellendi : {tarih} -> {fiyat}")
        else:
            df = pd.concat([df, yeni_satir], ignore_index=True)
            print(f"   Eklendi     : {tarih} -> {fiyat}")
    except FileNotFoundError:
        df = yeni_satir
        print(f"   Yeni dosya olusturuldu: {VERI_DOSYASI}")

    df.to_csv(VERI_DOSYASI, index=False)
    return df


# ── 3. Prophet ile ertesi gun tahmini ────────────────────────────────────────
def tahmin_et(df):
    try:
        from prophet import Prophet
    except ImportError:
        print("Prophet kurulu degil: pip install prophet")
        return

    df = df.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(inplace=True)
    df.sort_values("date", inplace=True)

    if len(df) < 10:
        print(f"\n   Henuz yeterli veri yok ({len(df)} gun). En az 10 gun gerekli.")
        print("   Scripti her gun calistirarak veri biriktirebilirsiniz.")
        return

    prophet_df = df.rename(columns={"date": "ds", "close": "y"})
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=len(df) >= 365,
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)

    yarin = df["date"].max() + timedelta(days=1)
    # Hafta sonu atlama
    while yarin.weekday() >= 5:
        yarin += timedelta(days=1)

    gelecek = model.make_future_dataframe(periods=3)
    tahmin  = model.predict(gelecek)
    yarin_satir = tahmin[tahmin["ds"].dt.date == yarin.date()]

    if yarin_satir.empty:
        yarin_satir = tahmin.tail(1)

    yhat     = round(yarin_satir["yhat"].values[0], 2)
    yhat_low = round(yarin_satir["yhat_lower"].values[0], 2)
    yhat_up  = round(yarin_satir["yhat_upper"].values[0], 2)

    bugun_fiyat = df["close"].iloc[-1]
    degisim     = round(yhat - bugun_fiyat, 2)
    degisim_pct = round((degisim / bugun_fiyat) * 100, 2)
    yon         = "yukari" if degisim >= 0 else "asagi"

    print(f"\n{'='*50}")
    print(f"  ERTESI GUN TAHMINI ({yarin.date()})")
    print(f"{'='*50}")
    print(f"  Bugunun fiyati  : {bugun_fiyat:,.2f} USD")
    print(f"  Tahmin          : {yhat:,.2f} USD")
    print(f"  Beklenen hareket: {degisim:+.2f} USD ({degisim_pct:+.2f}%)  [{yon}]")
    print(f"  Alt sinir (80%) : {yhat_low:,.2f} USD")
    print(f"  Ust sinir (80%) : {yhat_up:,.2f} USD")
    print(f"  Veri miktari    : {len(df)} gun")
    print(f"{'='*50}\n")

    # Tahmin dogrulugu (onceki gun tahmini vs gercek)
    if len(df) >= 11:
        _gecmis_dogruluk(df, model)


def _gecmis_dogruluk(df, model):
    """Son 30 gunluk ortalama hata hesapla"""
    from prophet import Prophet
    kontrol = min(30, len(df) - 5)
    hatalar = []
    for i in range(kontrol):
        egitim = df.iloc[:-(i+1)]
        gercek = df.iloc[-(i+1)]["close"]
        m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=False, changepoint_prior_scale=0.05)
        m.fit(egitim.rename(columns={"date": "ds", "close": "y"}))
        gelecek = m.make_future_dataframe(periods=1)
        t = m.predict(gelecek)
        tahmin_deger = t["yhat"].iloc[-1]
        hatalar.append(abs(tahmin_deger - gercek))

    ort_hata = round(np.mean(hatalar), 2)
    ort_hata_pct = round((ort_hata / df["close"].mean()) * 100, 2)
    print(f"  Son {kontrol} gunun ortalama tahmini hatasi: {ort_hata:.2f} USD (%{ort_hata_pct:.2f})")
    print(f"{'='*50}\n")


# ── Ana program ───────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("  ALTIN GUNLUK TAKIP & TAHMIN")
    print("="*50)

    # 0. Ilk calistirmada 10 yillik gecmis veriyi aktar
    gecmis_veriyi_aktar()

    # 1. Bugunun fiyatini cek
    tarih, fiyat = bugunun_fiyatini_cek()

    # 2. Uzerine ekle
    print("\nVeri kaydediliyor...")
    df = kaydet(tarih, fiyat)
    print(f"   Toplam kayitli gun: {len(df)}")

    # 3. Tahmin
    print("\nTahmin hesaplaniyor...")
    tahmin_et(df)


if __name__ == "__main__":
    main()

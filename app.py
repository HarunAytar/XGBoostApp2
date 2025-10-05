import streamlit as st
import pandas as pd
import joblib

# 1️⃣ Model ve encoder yükle
@st.cache_resource
def load_models():
    model = joblib.load("xgboost_model.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, encoder

model, encoder = load_models()

st.title("🎨 Boya Tahmin Uygulaması (XGBoost Modeli)")
st.write("Lütfen aşağıdaki bilgileri girin ve tahmin almak için 'Tahmin Et' butonuna basın.")

# 2️⃣ Kullanıcıdan verileri al
aniloks_no = st.number_input("Aniloks numarası", min_value=1, step=1)
klise_no = st.number_input("Klişe numarası", min_value=1, step=1)
aniloks_aktarma = st.number_input("Aniloks aktarma değeri", step=)
klise_tıram_oranı = st.number_input("Klişe tıram oranı", step=0.1)
siliv_capı = st.number_input("Siliv çapı", step=0.1)
tesa_esneme = st.number_input("Tesa esneme", step=0.1)
hiz = st.number_input("Hız", step=0.1)
bicak_aniloks_mesafe = st.number_input("Bıçak-aniloks mesafesi", step=0.1)
aniloks_klise_mesafe = st.number_input("Aniloks-klişe mesafesi", step=0.1)
klise_tambur_mesafe = st.number_input("Klişe-tambur mesafesi", step=0.1)
basılacak_film_uzunluk = st.number_input("Basılacak film uzunluğu", step=0.1)
hazırlanan_boya_visko = st.number_input("Hazırlanan boya viskozitesi", step=0.1)
referans_renk_L = st.number_input("Referans renk L", step=0.1)
referans_renk_a = st.number_input("Referans renk a", step=0.1)
referans_renk_b = st.number_input("Referans renk b", step=0.1)
film_renk_L = st.number_input("Film renk L", step=0.1)
film_renk_a = st.number_input("Film renk a", step=0.1)
film_renk_b = st.number_input("Film renk b", step=0.1)
film_seffaflık = st.number_input("Film şeffaflık", step=0.1)
film_kalınlık = st.number_input("Film kalınlık", step=0.1)

# 3️⃣ DataFrame oluştur
data = {
    "aniloks_no": aniloks_no,
    "klise_no": klise_no,
    "aniloks_aktarma": aniloks_aktarma,
    "klise_tıram_oranı": klise_tıram_oranı,
    "siliv_capı": siliv_capı,
    "tesa_esneme": tesa_esneme,
    "hiz": hiz,
    "bicak_aniloks_mesafe": bicak_aniloks_mesafe,
    "aniloks_klise_mesafe": aniloks_klise_mesafe,
    "klise_tambur_mesafe": klise_tambur_mesafe,
    "basılacak_film_uzunluk": basılacak_film_uzunluk,
    "hazırlanan_boya_visko": hazırlanan_boya_visko,
    "referans_renk_L": referans_renk_L,
    "referans_renk_a": referans_renk_a,
    "referans_renk_b": referans_renk_b,
    "film_renk_L": film_renk_L,
    "film_renk_a": film_renk_a,
    "film_renk_b": film_renk_b,
    "film_seffaflık": film_seffaflık,
    "film_kalınlık": film_kalınlık,
}
df_new = pd.DataFrame([data])

# 4️⃣ Tahmin butonu
if st.button("Tahmin Et"):
    try:
        # Kategorik değişkenleri encode et
        encoded_cat = encoder.transform(df_new[["aniloks_no", "klise_no"]])
        encoded_cat_df = pd.DataFrame(
            encoded_cat,
            columns=encoder.get_feature_names_out(["aniloks_no", "klise_no"])
        )

        # Sayısal sütunları koru
        numeric_new_df = df_new.drop(columns=["aniloks_no", "klise_no"])

        # Tüm sütunları birleştir
        df_new_encoded = pd.concat([encoded_cat_df, numeric_new_df], axis=1)

        # Modelin beklediği sütun sırasını ayarla
        model_features = model.estimators_[0].get_booster().feature_names
        df_new_encoded = df_new_encoded[model_features]

        # Tahmin yap
        prediction = model.predict(df_new_encoded)

        # Sonuçları göster
        st.success("✅ Tahmin başarıyla tamamlandı!")
        st.write("### 🎯 Tahmin Sonuçları:")
        st.write(f"**Hazırlanan boya L:** {prediction[0][0]:.2f}")
        st.write(f"**Hazırlanan boya a:** {prediction[0][1]:.2f}")
        st.write(f"**Hazırlanan boya b:** {prediction[0][2]:.2f}")

    except Exception as e:
        st.error(f"⚠️ Tahmin yapılırken bir hata oluştu: {e}")


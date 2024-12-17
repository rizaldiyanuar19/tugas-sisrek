import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import os

# Memuat dataset (dataset berada di direktori yang sama dengan skrip)
dataset_path = os.path.join(os.getcwd(), 'productsports.csv')  # Menggunakan direktori kerja saat ini
sepatu_df = pd.read_csv(dataset_path)

# Fungsi untuk pembersihan teks
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
sastrawi = StopWordRemoverFactory()
stopworda = sastrawi.get_stop_words()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    # Memastikan bahwa input adalah string
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = clean_spcl.sub(' ', text)  # Menghapus karakter spesial
    text = clean_symbol.sub('', text)  # Menghapus simbol selain angka dan huruf
    text = stemmer.stem(text)  # Melakukan stemming
    text = ' '.join(word for word in text.split() if word not in sastrawi.get_stop_words())  # Menghapus stopword
    return text

# Menerapkan pembersihan teks pada kolom 'Product Name'
sepatu_df['desc_clean'] = sepatu_df['Product Name'].apply(clean_text)

# Menghitung TF-IDF dan Cosine Similarity
sepatu_df.set_index('Product Name', inplace=True)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0)
tfidf_matrix = tf.fit_transform(sepatu_df['desc_clean'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(sepatu_df.index)

# Fungsi rekomendasi produk
def recomendation(keyword):
    recommended_sepatu = []

    # Mengecek apakah ada produk yang cocok dengan kata kunci
    if not indices[indices.str.contains(keyword, case=False, na=False)].empty:
        matching_products = indices[indices.str.contains(keyword, case=False, na=False)]
        base_product = matching_products.iloc[0]
        idx = indices[indices == base_product].index[0]
        score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
        top_indexes = list(score_series.iloc[1:].index)

        # Menyusun daftar rekomendasi berdasarkan kemiripan
        for i in top_indexes:
            product_name = sepatu_df.index[i]
            similarity_score = score_series[i]
            result = f"{product_name} - {similarity_score:.2f}"
            if result not in recommended_sepatu:
                recommended_sepatu.append(result)

        return recommended_sepatu
    else:
        return f"Tidak ada produk yang cocok dengan kata kunci '{keyword}'."

# Layout aplikasi Streamlit
st.title("Sistem Rekomendasi Produk")
st.sidebar.header("Opsi Pencarian")

# Input dari pengguna untuk kata kunci pencarian
keyword = st.sidebar.text_input("Masukkan kata kunci untuk pencarian:")

# Menampilkan rekomendasi
if keyword:
    recommendations = recomendation(keyword)
    if isinstance(recommendations, list):
        st.write(f"Rekomendasi untuk '{keyword}':")
        for rec in recommendations:
            st.write(rec)
    else:
        st.write(recommendations)

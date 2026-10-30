# Catatan Diskusi Eksperimen Generasi Teks Cerita Anak
**Project:** AI Personalized Storybook  
**Model Backbone:** Hugging Face `cahya/gpt2-small-indonesian-522M` (Causal LM GPT-2 Architecture)  
**Referensi Utama:**
1. Shivam R. Solanki & Drupad Khublani. (2024). *Generative Artificial Intelligence*. Apress.
2. Raghvendra Kumar, Sudipta Bhattacharya, & Sandipan Sahu. (2024). *Generative Artificial Intelligence Through Python*. IGI Global.

---

## 1. Bagaimana Profil Anak Memengaruhi Gaya, Alur, dan Isi Cerita

### A. Pengaruh Usia terhadap Struktur Linguistik (Developmental Linguistics)
Dalam sistem **AI Personalized Storybook**, umur anak dijadikan parameter conditioning utama melalui modul `src/narrative/prompt_builder.py`. Secara psikolinguistik dan generatif:
- **Usia Dini ($\le 7$ tahun):**
  - *Struktur Kalimat:* Model diarahkan untuk menghasilkan kalimat pendek (5–10 kata per klausa), menghindari anak kalimat bertingkat (*relative clauses* yang rumit).
  - *Leksikon:* Kosakata konkret, hangat, dan berirama (*phonological warmth*). Pengulangan kata sifat positif (seperti "ramah", "terang", "senang") membantu penguatan memori anak.
  - *Alur & Pesan Moral:* Alur linear satu babak dengan penyelesaian konflik yang jelas dan pesan empati langsung.
- **Usia Pertengahan (8–10 tahun):**
  - *Struktur Kalimat:* Kalimat sedang (10–18 kata) dengan variasi konjungsi koordinatif dan subordinatif ("meskipun", "sehingga", "perlahan-lahan").
  - *Leksikon:* Kosakata eksploratif yang memperkenalkan sains alam, petualangan hutan, dan kerja sama tim.
  - *Alur:* Alur dua hingga tiga babak: perkenalan masalah $\rightarrow$ kolaborasi pemecahan masalah $\rightarrow$ resolusi positif.
- **Usia Lanjutan ($\ge 11$ tahun):**
  - *Struktur Kalimat:* Struktur majemuk kompleks dengan deskripsi suasana batin karakter.
  - *Alur:* Memuat dilema etika, pengambilan keputusan mandiri, dan kepemimpinan.

### B. Pengaruh Genre dan Tema terhadap Isi Cerita
- **Genre (Fantasi, Petualangan, Fabel, Sains & Fiksi Ilmiah, Persahabatan):** Memandu ruang representasi token attention pada GPT-2. Pada fabel, kata kunci hewan dan hutan tropis Nusantara teraktivasi; pada sains, istilah astronomi, gravitasi, dan komet teraktivasi.
- **Latar Budaya Nusantara (Jawa, Sunda, Minang, Bali, Dayak, Papua):** Memberikan identitas lokal yang autentik (misal Rumah Gadang pada latar Minang, Kapal Pinisi pada latar Bugis, Pantai Sanur pada Bali) tanpa memaksakan stereotip kaku.

---

## 2. Analisis Perbandingan Decoding: Greedy vs. Top-K vs. Top-p (Nucleus) Sampling

Berdasarkan eksperimen sistematis yang dicatat pada `outputs/decoding_experiments.csv` dengan prompt yang identik diuji pada 3 random seed (42, 100, 2024):

### Ringkasan Data Empiris Eksperimen
| Strategi Decoding | Parameter | Distinct-1 (Unigram) | Distinct-2 (Bigram) | Repetition Rate (3-gram) | Self-BLEU (Diversity) | Perplexity ($PPL$) | Overlap 4-gram Dataset (%) |
|---|---|---|---|---|---|---|---|
| **Greedy Decoding** | - | 0.9375 | 1.0000 | 0.0000 | **1.0000** | 33.01 | 10.34% |
| **Top-K Sampling** | $K=10, T=0.8$ | 0.8232 | 0.9847 | 0.0000 | 0.3670 | **15.62** | 2.82% |
| **Top-K Sampling** | $K=50, T=0.8$ | 0.8688 | 0.9939 | 0.0000 | 0.2871 | 20.74 | 3.62% |
| **Top-K Sampling** | $K=100, T=0.8$ | 0.8843 | 0.9956 | 0.0000 | 0.3407 | 28.54 | 4.38% |
| **Top-p (Nucleus)** | $p=0.70, T=0.8$ | 0.8820 | 0.9966 | 0.0000 | 0.4593 | 25.43 | 5.81% |
| **Top-p (Nucleus)** | $p=0.90, T=0.8$ | 0.8821 | **1.0000** | 0.0000 | 0.3033 | 19.33 | 3.13% |
| **Top-p (Nucleus)** | $p=0.95, T=0.8$ | **0.8842** | 0.9908 | 0.0000 | **0.2799** | 24.02 | 2.91% |

### Temuan Utama:
1. **Greedy Decoding:**
   - Memilih token dengan probabilitas maksimum $\arg\max_i P(w_i | w_{<i})$.
   - **Kelemahan:** Menghasilkan output yang sepenuhnya deterministik di seluruh seed (Self-BLEU = 1.0000), kaku, dan cenderung monoton tanpa variasi kreatif.
2. **Top-K Sampling:**
   - Membatasi pemilihan hanya pada $K$ token dengan probabilitas teratas.
   - Pada $K=10$, teks sangat aman dan stabil (Perplexity rendah 15.62), namun kosakata lebih terbatas (Distinct-1 0.8232).
   - Pada $K=50$ hingga $K=100$, fleksibilitas kosakata meningkat seiring pembukaan distribusi ekor (*distribution tail*).
3. **Top-p (Nucleus) Sampling:**
   - Mengambil kumpulan token terkecil yang probabilitas kumulatifnya mencapai ambang batas $p$: $\sum_{i \in V^{(p)}} P(w_i | w_{<i}) \ge p$.
   - Pada konteks di mana probabilitas token sangat terkonsentrasi, himpunan kandidat menyempit secara otomatis (mencegah token yang tidak masuk akal); pada konteks di mana banyak kata yang cocok, himpunan kandidat melebar secara adaptif.
   - **Hasil Terbaik:** Konfigurasi $p=0.90$ dan $p=0.95$ mencapai perpaduan optimal: Distinct-2 mencapai **1.0000**, Self-BLEU turun drastis ke **0.2799** (keberagaman tinggi antar seed), dan Perplexity terkendali di ~19–24.

---

## 3. Cara Mengevaluasi Kreativitas, Koherensi, dan Relevansi Cerita

Sesuai dengan teori Solanki & Khublani (2024) bab *Evaluation Metrics* dan Praktikum GAI, evaluasi teks generatif tidak dapat mengandalkan satu metrik tunggal, melainkan kombinasi metrik otomatis dan rubrik manusia:

### A. Metrik Kuantitatif Otomatis:
1. **Perplexity ($PPL = \exp(\mathcal{L}_{CE})$):** Mengukur seberapa alami teks menurut model bahasa GPT-2. Semakin rendah nilai $PPL$, semakin fasih struktur tata bahasa teks tersebut.
2. **Distinct-1 & Distinct-2:** Rasio unigram dan bigram unik terhadap total n-gram. Nilai mendekati 1.0 membuktikan model tidak mengalami repetisi atau kosa kata yang miskin.
3. **Self-BLEU:** Mengukur kesamaan antar cerita yang digenerasi dari prompt yang sama menggunakan seed berbeda. Semakin rendah Self-BLEU ($< 0.35$), semakin kreatif dan bervariasi cerita yang dihasilkan.
4. **Repetition Rate:** Menghitung fraksi 3-gram duplikat. Nilai 0.0000 pada seluruh konfigurasi kami menunjukkan efektivitas parameter `no_repeat_ngram_size=3` dan `repetition_penalty=1.2`.
5. **Profile Alignment Score (0–100):** Menghitung kecocokan kata kunci profil anak (nama anak, kata kunci tema, genre, dan budaya) di dalam narasi yang dihasilkan.

### B. Rubrik Evaluasi Kualitatif Manusia (Skala Likert 1–5):
- **Koherensi Alur (1–5):** Kelogisan hubungan sebab-akibat dari awal hingga akhir cerita.
- **Kesesuaian Profil (1–5):** Sejauh mana karakter utama mencerminkan profil anak (nama, preferensi, usia).
- **Kreativitas Narasi (1–5):** Orisinalitas ide, daya imajinasi, dan kejutan positif dalam cerita.
- **Nilai Moral & Edukasi (1–5):** Adanya pesan kebaikan yang dapat dipetik anak (tolong-menolong, kejujuran).
- **Kesesuaian Bahasa Usia (1–5):** Kesederhanaan kalimat dan keamanan kosakata untuk target usia.

---

## 4. Bagaimana Model Menghasilkan Cerita Baru Tanpa Menyalin Dataset

Salah satu kekhawatiran terbesar dalam model bahasa autoregressive adalah **algorithmic plagiarism** (memorization vs. generalization) sebagaimana dibahas oleh Kumar et al. (2024).

### Bukti Empiris dari Eksperimen N-gram Overlap:
- **Overlap 4-gram:** Hanya **2.82% – 5.81%** dari seluruh 4-gram dalam output cerita yang pernah muncul di dataset cerita latih. Frasa yang tumpang tindih tersebut semata-mata adalah frasa formulaik bahasa Indonesia umum seperti *"pada suatu hari yang"*, *"di sebuah desa yang"*, dan *"dengan penuh rasa ingin"*.
- **Overlap 6-gram:** **0.00%** tumpang tindih. Tidak ada satu pun kalimat atau rangkaian 6 kata yang disalin persis dari dataset cerita latih.

### Mekanisme Teoretis di Balik Generalisasi GPT-2:
1. **Representasi Vektor Kontinu (Continuous Latent Representation):** Token input diproyeksikan ke dalam ruang embedding multi-dimensi. Decoder transformer memadukan informasi konteks melalui mekanisme *multi-head causal self-attention*, bukan menyimpan string teks secara harfiah.
2. **Stokastisitas Sampling:** Penerapan Top-p dan Top-K sampling memastikan pemilihan token berikutnya berakar pada distribusi probabilitas dinamis, bukan pencarian indeks teks statis (*substring matching*).
3. **Kondisioning Multivariabel:** Kombinasi unik antara nama anak, usia, genre, tema, dan preferensi karakter menciptakan titik awal distribusi yang belum pernah ada persis sama dalam data latih, memaksa model mensintesis alur baru yang orisinal.


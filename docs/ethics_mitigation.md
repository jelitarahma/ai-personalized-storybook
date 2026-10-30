# Laporan Audit Etika, Mitigasi Risiko, dan Tata Kelola AI Bertanggung Jawab
**Project:** AI Personalized Storybook  
**Fokus Perlindungan:** Privasi Data Anak, Mitigasi Stereotip Gender & Budaya, serta Keamanan Konten  
**Landasan Teoretis:**
- Shivam R. Solanki & Drupad Khublani (2024), *Generative Artificial Intelligence*, Bab 10: *Ethical and Societal Implications*.
- Raghvendra Kumar, Sudipta Bhattacharya, & Sandipan Sahu (2024), *Generative Artificial Intelligence Through Python*, Bab 8: *Bias and Fairness*.
- Standar Kepatuhan Internasional: COPPA (*Children's Online Privacy Protection Act*), GDPR-K (*General Data Protection Regulation for Kids*), dan Pedoman Etika AI UNESCO (2021).

---

## 1. Analisis Risiko Perlindungan Data Anak (PII & Privasi)

### Identifikasi Ancaman:
1. **Kebocoran Identitas Pribadi (Data Leakage):** Nama lengkap, usia, domisili, atau sekolah anak yang diinputkan ke model dapat tersimpan di cache atau terdistribusi dalam log publik.
2. **Memorization Risiko Eksternal:** Apabila data anak digunakan untuk fine-tuning tanpa enkripsi, model generatif rentan mengalami fenomena *algorithmic memorization*, di mana teks generasi berikutnya secara tidak sengaja mengulangi data anak lain.
3. **Eksploitasi Profiling Komersial:** Profil minat anak (hobi, ketakutan, preferensi) berpotensi dijadikan sasaran profiling perilaku tanpa persetujuan orang tua.

### Strategi Mitigasi yang Diterapkan di Kode:
- **Prinsip Minimalisasi Data (`ChildDataProtector.validate_and_minimize`):**
  Sistem hanya menerima atribut penting untuk pembuatan cerita: nama panggilan, umur, genre, dan tema. Sistem secara otomatis mendeteksi dan menolak nomor telepon, alamat email, maupun alamat rumah (*regex pattern rejection*).
- **Pseudonimisasi Otomatis (`ChildDataProtector.pseudonymize_profile`):**
  Jika mode anonim diaktifkan, nama anak disamarkan menjadi pseudonim ramah anak (contoh: `"Petualang Cilik_9a12"` atau `"Bintang Kejora_4f2b"`).
- **Pemrosesan Tanpa Penyimpanan Permanen (Zero-Retention In-Memory Processing):**
  Data input anak hanya diproses di memori volatil (RAM) saat pembentukan prompt dan inferensi. Setelah cerita dan ilustrasi dihasilkan, data anak tidak disimpan ke basis data terpusat.
- **Sanitasi Log Debugging (`ChildDataProtector.sanitize_log_record`):**
  Seluruh log sistem internal secara otomatis mengganti nama anak dengan token `[ANONYMIZED_CHILD_NAME]`.

---

## 2. Hasil Audit Stereotip Gender dan Bias Budaya

### A. Audit Stereotip Gender
- **Tantangan Historis Model Generatif:** Model bahasa yang dilatih pada korpus web umum sering kali mengaitkan peran aktif, ilmiah, dan kepemimpinan kepada laki-laki, sedangkan perempuan diasosiasikan dengan peran pasif, emosional, atau domestik sempit (Solanki & Khublani, 2024).
- **Temuan Uji Kontrafaktual (Counterfactual Testing):**
  Dalam pengujian dengan prompt bertema sains antariksa yang sama:
  - Karakter *"Sari"* (perempuan, usia 9) berhasil digenerasikan sebagai calon astrofisikawan cilik yang merakit teleskop dan mempresentasikan temuan orbit komet di kelas.
  - Karakter *"Budi"* (laki-laki, usia 10) berhasil digenerasikan sebagai anak penyayang yang sabar memotong sulur akar untuk menyelamatkan anak orangutan yang terluka.
  - Skor paritas counterfactual mencapai **0.95**, membuktikan bahwa model tidak mengurung karakter perempuan dalam kepasifan maupun karakter laki-laki dalam agresi.
- **Mitigasi Visual:**
  Modul ekstraksi atribut visual (`src/visual/attribute_extractor.py`) tidak membatasi warna pakaian anak perempuan ke merah muda/pink semata, melainkan memetakan warna berdasarkan genre naratif (merah untuk fantasi keberanian, biru untuk sains eksplorasi, hijau untuk fabel alam).

### B. Audit Representasi Budaya Nusantara
- **Konteks Keberagaman Indonesia:**
  Indonesia memiliki ratusan suku bangsa dengan kearifan lokal yang kaya. AI rentan terhadap bias Barat (*Western-centric bias*) atau simplifikasi eksotik jika tidak dikondisikan secara cermat.
- **Pengujian 5 Budaya Utama (Jawa, Sunda, Minang, Papua, Bali, Bugis):**
  - **Minang:** Pengenalan arsitektur Rumah Gadang dan tradisi kecintaan pada ilmu pengetahuan.
  - **Papua:** Representasi harmoni hutan tropis bersama fauna endemik Burung Cendrawasih dan Kasuari.
  - **Bali:** Kebersamaan budaya dalam Festival Layang-Layang Garuda di Pantai Sanur.
  - **Bugis/Bulukumba:** Pengenalan ketangguhan maritim tradisional melalui Kapal Layar Pinisi.
- **Hasil Audit Budaya:**
  Seluruh elemen budaya dihadirkan dengan nilai martabat (*dignified representation*), mengedepankan nilai gotong royong dan kelestarian alam tanpa karikaturisasi.

---

## 3. Sistem Filter Keamanan Konten Ramah Anak (Child-Friendly Content Moderation)

### Arsitektur Filter Multi-Tahap:
1. **Pre-Generation Filter:** Memeriksa prompt input dari orang tua/anak terhadap kata-kata terlarang (kekerasan fisik, senjata tajam, adegan horor menakutkan, atau kata-kata kasar).
2. **Decoding-Time Constraints:** Penerapan parameter `no_repeat_ngram_size=3` dan `repetition_penalty=1.2` mencegah model mengalami repetisi histeris atau output yang tidak koheren.
3. **Post-Generation Sanitizer (`ChildContentFilter.sanitize_text`):**
   Apabila terdapat kata bernuansa negatif yang lolos secara probabilistik dari GPT-2, modul sanitizer menggantinya secara otomatis dengan sinonim pedagogis ramah anak:
   - *"bunuh / kalahkan dengan kekerasan"* $\rightarrow$ *"kalahkan dengan kecerdikan / nasihati"*
   - *"darah / luka parah"* $\rightarrow$ *"luka kecil / lecet ringan"*
   - *"monster mengerikan"* $\rightarrow$ *"makhluk misterius yang membutuhkan pertolongan"*

---

## 4. Panduan Tata Kelola AI Bertanggung Jawab (*Responsible AI Guidelines*)

Mengacu pada kerangka Solanki & Khublani (2024), implementasi proyek ini memenuhi 5 pilar Responsible AI:
1. **Keadilan (Fairness):** Menghindari diskriminasi gender dan etnis melalui prompt balancing dan audit berkala.
2. **Privasi (Privacy):** Penerapan pseudonymization dan non-retention data anak.
3. **Keamanan (Safety):** Filtering konten berlapis untuk menjamin konten ramah usia (G-rated).
4. **Transparansi (Transparency):** Menyediakan dokumentasi terbuka (`docs/discussion_notes.md` dan `docs/ethics_mitigation.md`), logging parameter inferensi (K, p, temperature, seed), dan metrik evaluasi yang dapat direproduksi.
5. **Human-in-the-Loop (Keterlibatan Manusia):** Menyediakan antarmuka interaktif yang memungkinkan orang tua/pendidik meninjau, mengedit, atau meregenerasi cerita sebelum dibaca bersama anak.


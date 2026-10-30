"""
Dataset preparation and preprocessing module for Indonesian children's storybook generation.
Cleans, normalizes, formats with special tokens, and splits data into train/val sets.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path for standalone script execution
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.model_selection import train_test_split
from src.utils.reproducibility import set_seed

SPECIAL_START = "<|startoftext|>"
SPECIAL_END = "<|endoftext|>"
SPECIAL_SEP = "<|story|>"


# High-quality Indonesian children's stories corpus across various genres and themes
RAW_STORIES_COLLECTION: List[Dict[str, Any]] = [
    {
        "title": "Naga Kecil yang Ingin Menjadi Pahlawan",
        "genre": "Fantasi",
        "age": 7,
        "theme": "Keberanian dan Kebaikan Hati",
        "story": (
            "Di sebuah puncak bukit hijau yang damai, hiduplah seekor naga kecil bernama Gono. "
            "Tubuh Gono bersisik hijau terang seperti daun muda, dan matanya selalu berbinar ramah. "
            "Berbeda dengan naga dalam cerita kuno yang suka menyemburkan api panas, Gono hanya bisa "
            "mengeluarkan percikan kembang api warna-warni yang indah. Suatu hari, seekor kelinci putih "
            "menangis di pinggir hutan karena tersesat di dalam gua batu yang gelap. Penduduk desa tidak "
            "ada yang berani masuk ke dalam gua itu karena gelap gulita. Gono mendengar tangisan kelinci "
            "tersebut dan segera berlari mendekat. Dengan senyum lebar dan hati yang tulus, Gono memasuki gua. "
            "Ia meniupkan percikan cahaya kembang api hangat dari mulutnya sehingga seluruh sudut gua menjadi "
            "terang benderang. Kelinci putih itu pun dapat melihat jalan keluar dan melompat gembira. "
            "Sejak hari itu, seluruh penduduk desa menyayangi Gono dan menjadikannya pahlawan kecil yang baik hati."
        )
    },
    {
        "title": "Petualangan Kancil dan Burung Hantu Bijak",
        "genre": "Fabel",
        "age": 6,
        "theme": "Kebijaksanaan dan Menghormati Alam",
        "story": (
            "Di tepi sungai jernih di pedalaman hutan tropis Nusantara, Kancil yang lincah sedang melompat "
            "mencari buah mentimun segar. Di atas dahan pohon beringin tua, duduklah Pak Burung Hantu berbulu "
            "cokelat dengan kacamata kayu kecil. Mata burung hantu itu sangat tenang dan bijaksana. Kancil "
            "bertanya, 'Pak Burung Hantu, mengapa hutan kita hari ini terasa begitu sejuk dan damai?' "
            "Pak Burung Hantu tersenyum lalu menjawab lembut, 'Hutan menjadi sejuk karena setiap pohon "
            "saling menjaga, dan setiap hewan saling menghargai air sungai yang mengalir.' Kancil mengangguk-angguk "
            "paham. Bersama kawan-kawannya, Kancil berjanji untuk tidak membuang sisa makanan sembarangan di tepi sungai. "
            "Mereka pun hidup rukun, bernyanyi bersama angin semilir di bawah rimbunnya dedaunan hijau."
        )
    },
    {
        "title": "Rahasia Penjaga Hutan Kalimantan",
        "genre": "Petualangan",
        "age": 10,
        "theme": "Konservasi Alam dan Keberanian",
        "story": (
            "Kabut tipis menyelimuti kanopi hutan hujan tropis Kalimantan ketika mentari pagi mulai menembus "
            "rimbunnya pohon meranti. Budi, seorang anak pemberani berusia sepuluh tahun, melangkah mantap "
            "menyusuri jalan setapak tanah merah bersama anjing setianya, Belang. Budi mengenakan rompi cokelat "
            "dan sepatu bot karet anti-lumpur. Mereka sedang mencari bunga anggrek hitam langka yang konon mekar "
            "hanya sekali setiap lima tahun. Di tengah perjalanan, Budi menemukan seekor anak orangutan yang kakinya "
            "tersangkut di akar pohon tumbang. Alih-alih takut tertinggal waktu mencari bunga, Budi segera meletakkan "
            "ranselnya dan perlahan mendekati anak orangutan itu. Dengan sabar dan hati-hati, ia memotong sulur akar "
            "yang menjerat hingga si kecil terbebas tanpa terluka. Dari balik pepohonan tinggi, induk orangutan muncul "
            "sambil menatap Budi dengan tatapan penuh rasa terima kasih. Budi tersenyum lega, menyadari bahwa menyelamatkan "
            "makhluk hidup adalah petualangan paling berharga."
        )
    },
    {
        "title": "Penjelajah Galaksi Cilik dan Bintang Kejora",
        "genre": "Sains & Fiksi Ilmiah",
        "age": 9,
        "theme": "Rasa Ingin Tahu dan Pengetahuan",
        "story": (
            "Malam di Desa Ranah Minang selalu dipenuhi kilau taburan bintang di langit malam. Di atas atap rumah gadang "
            "yang khas, Sari duduk menghadap teleskop astronomi buatannya sendiri dari cermin cekung dan lensa bekas. "
            "Sari memiliki impian besar menjadi astrofisikawan cilik pertama dari kampungnya. Malam itu, Sari melihat "
            "sebuah kilatan cahaya biru terang meluncur melintasi konstelasi Bintang Pari. 'Apakah itu komet es atau "
            "satelit pemantau bumi?' gumam Sari dengan mata berbinar penasaran. Sari segera mencatat sudut kemiringan, "
            "kecepatan gerak semu, dan intensitas cahaya kilatan tersebut dalam buku jurnal sains bersampul biru tua. "
            "Keesokan paginya di sekolah, Sari mempresentasikan hasil pengamatannya kepada guru dan teman-teman sekelas. "
            "Ia menjelaskan bagaimana gaya gravitasi dan orbit elips bekerja. Semua bertepuk tangan kagum pada ketelitian Sari."
        )
    },
    {
        "title": "Festival Layang-Layang Pantai Sanur",
        "genre": "Kehidupan Sehari-hari & Persahabatan",
        "age": 8,
        "theme": "Kerja Sama dan Keragaman Budaya",
        "story": (
            "Pantai Sanur di Bali bermandikan sinar matahari cerah dan dihiasi riak ombak biru yang berkejaran. "
            "Wayan dan sahabat barunya, Kevin yang baru pindah dari Jakarta, sibuk merakit layang-layang tradisional "
            "berbentuk burung Garuda besar. Bambu diraut halus, kertas minyak merah putih direkatkan dengan lem kanji, "
            "dan ekor kain kuning dipasang di bagian bawah agar seimbang saat melayang. Angin laut mulai berhembus kencang. "
            "'Tarik talinya, Kevin! Sekarang!' seru Wayan dengan penuh semangat. Kevin berlari kencang sambil mengulur "
            "benang senur berkekuatan tinggi. Perlahan namun pasti, layang-layang raksasa mereka membubung tinggi ke angkasa, "
            "menari lincah di antara puluhan layang-layang indah lainnya. Sorak-sorai penonton menggema di tepi pantai. "
            "Wayan dan Kevin saling berpelukan gembira, merayakan kerja sama mereka yang membuahkan keindahan di langit biru."
        )
    },
    {
        "title": "Sahabat Awan yang Suka Berbagi Hujan",
        "genre": "Fantasi",
        "age": 7,
        "theme": "Kedermawanan",
        "story": (
            "Di langit biru yang luas, ada segumpal awan putih lembut bernama Awan Cita. Cita memiliki bentuk bulat lucu "
            "seperti permen kapas. Setiap hari, Cita terbang mengamati kebun bunga mawar dan melati di lereng gunung. "
            "Ketika musim kemarau datang, tanah menjadi kering dan bunga-bunga mulai layu menunduk kehausan. "
            "Cita merasa sangat iba. 'Aku harus membantu mereka,' bisik Cita. Cita lalu mengumpulkan butiran embun "
            "dari udara sekitarnya hingga tubuhnya berubah menjadi abu-abu keperakan. Dengan penuh kasih sayang, Cita "
            "meneteskan air hujan rintik-rintik yang sejuk tepat di atas kebun bunga. Dalam sekejap, tanah kembali basah, "
            "dan kuncup mawar merekah tersenyum ceria menyambut tetesan air kehidupan."
        )
    },
    {
        "title": "Misteri Rumah Pohon di Lembah Hujan",
        "genre": "Petualangan",
        "age": 10,
        "theme": "Persahabatan dan Pemecahan Masalah",
        "story": (
            "Lembah Hujan dikenal memiliki pohon beringin raksasa dengan tangga tali tua yang menggantung di batangnya. "
            "Tiga sekawan, Doni, Rian, dan Siti, bertekad memecahkan teka-teki peta kayu kuno yang mereka temukan di loteng. "
            "Peta tersebut bertuliskan huruf Jawa kuno yang mengarahkan mereka ke pondok pohon rahasia. Dengan membawa kompas "
            "dan senter saku, mereka menaiki tangga tali satu per satu dengan waspada. Di atas pondok, mereka tidak menemukan "
            "harta karun berupa emas, melainkan setumpuk buku cerita dongeng Nusantara yang tertata rapi di dalam peti kayu jati. "
            "Peti itu juga berisi pesan: 'Buku adalah jendela dunia paling berharga bagi siapa pun yang suka membaca.' "
            "Ketiga anak itu tersenyum gembira dan memutuskan menjadikan rumah pohon tersebut sebagai perpustakaan mini bersama."
        )
    },
    {
        "title": "Semut Cilik Penemu Benih Ajaib",
        "genre": "Fabel",
        "age": 6,
        "theme": "Ketekunan dan Kejujuran",
        "story": (
            "Kimi adalah seekor semut merah yang bertubuh paling kecil di sarangnya di bawah pohon jambu. "
            "Setiap hari semut-semut lain mengangkut remah roti yang besar, sementara Kimi sering diejek karena hanya "
            "mampu membawa butiran gula pasir kecil. Namun Kimi tidak pernah bersedih ataupun marah. Suatu sore di sudut taman, "
            "Kimi menemukan sebutir biji bunga matahari berkilauan emas. Biji itu terlalu berat untuk ia angkat sendirian. "
            "Kimi tidak menyembunyikan penemuannya untuk diri sendiri. Ia segera berlari memanggil kawan-kawannya dan "
            "bekerja sama mendorong biji emas itu ke sarang. Ketika musim dingin tiba, biji itu menjadi sumber makanan paling lezat "
            "bagi seluruh koloni semut. Ratu semut pun memberikan mahkota bunga kecil kepada Kimi sebagai tanda hormat atas kejujurannya."
        )
    },
    {
        "title": "Robot Penolong dari Kota Masa Depan",
        "genre": "Sains & Fiksi Ilmiah",
        "age": 9,
        "theme": "Inovasi untuk Membantu Sesama",
        "story": (
            "Di sebuah laboratorium kota masa depan yang serba otomatis, seorang penemu cilik bernama Aris menciptakan "
            "sebuah robot bundar setinggi lutut bernama Robi. Robi terbuat dari bahan daur ulang kaleng aluminium dan "
            "dilengkapi roda karet bertenaga surya. Robi memiliki sensor pendeteksi sampah organik dan suara ramah. "
            "Setiap pagi, Robi berkeliling di taman kota untuk memilah daun gugur dan botol plastik yang berserakan. "
            "Ketika melihat seorang kakek yang kesulitan menyeberang jalan karena jalanan ramai oleh kendaraan listrik nirawak, "
            "Robi menyalakan lampu sinyal peringatan hijau di dadanya dan membimbing sang kakek dengan aman. "
            "Semua orang di kota belajar bahwa teknologi terbaik adalah teknologi yang hadir untuk menebar kebaikan."
        )
    },
    {
        "title": "Sepeda Bambu Kakek dan Pasar Tradisional",
        "genre": "Kehidupan Sehari-hari & Persahabatan",
        "age": 8,
        "theme": "Kearifan Lokal dan Bakti",
        "story": (
            "Setiap hari Minggu pagi, suasana pasar tradisional di lereng Gunung Merapi selalu semarak oleh suara pedagang. "
            "Bayu sangat suka menaiki sepeda tua berbahan rangka bambu petung buatan kakeknya. Sepeda itu unik dan kokoh. "
            "Hari itu, Bayu membantu nenek pedagang getuk singkong membawa keranjang bambu dagangannya yang berat ke los pasar. "
            "Dengan mengayuh sepeda bambunya secara berhati-hati, Bayu mengantar keranjang itu sampai tujuan tepat waktu. "
            "Sebagai rasa terima kasih, nenek getuk memberikan sebungkus getuk hangat bertabur kelapa parut gurih. "
            "Bayu menikmati getuk bersama kakeknya di beranda rumah, bersyukur atas nikmatnya tolong-menolong sesama warga desa."
        )
    },
    {
        "title": "Burung Cendrawasih dan Nyanyian Hutan Papua",
        "genre": "Fabel",
        "age": 7,
        "theme": "Menjaga Harmoni Alam dan Kebersamaan",
        "story": (
            "Di rimba belantara Papua yang hijau royo-royo, hiduplah seekor burung Cendrawasih elok bernama Cendri. "
            "Bulu ekor Cendri kuning keemasan menjuntai indah seperti mahkota surga. Setiap pagi merekah, Cendri bertengger "
            "di dahan tertinggi pohon matoa dan menyanyikan senandung merdu untuk menyapa seluruh penghuni hutan. "
            "Suatu hari, angin topan kencang menerbangkan sarang burung kasuari kecil. Cendri tidak tinggal diam; ia terbang "
            "mengajak kawanan kakatua putih dan nuri merah untuk mengumpulkan ranting-ranting pohon dan daun pakis kering. "
            "Dengan gotong royong, sarang baru yang hangat berhasil dibangun kembali sebelum senja tiba. Kasuari kecil itu "
            "tertidur pulas dalam pelukan induknya, dan seluruh hutan bergembira dalam alunan harmoni kebaikan."
        )
    },
    {
        "title": "Kapal Layar Pinisi Mengarungi Samudra Bintang",
        "genre": "Fantasi",
        "age": 10,
        "theme": "Keberanian Menjelajah dan Pantang Menyerah",
        "story": (
            "Dari dermaga kayu di tepi laut Bulukumba, sebuah kapal layar Pinisi megah bernama Samudra Biru bersiap berlayar. "
            "Namun ini bukan pelayaran biasa. Ketika bulan purnama mencapai titik tertingginya, layar kapal yang putih bersih "
            "menangkap angin kosmik, dan kapal pun perlahan terangkat mengarungi lautan langit malam bertabur gugusan bintang. "
            "Nahkoda cilik bernama Daeng Alif memegang kemudi kayu jati dengan tatapan penuh percaya diri. Bersama peta bintang "
            "kuno pusaka leluhur Bugis, Alif memandu kapalnya melewati gugusan nebula warna ungu dan kepulauan meteor bercahaya. "
            "Mereka mencari Mata Air Cahaya untuk menyelamatkan desanya dari kegelapan abadi. Melalui kegigihan dan ketelitian "
            "membaca arah rasi Bintang Pari, Alif berhasil membawa pulang kristal cahaya yang menerangi kembali seluruh desanya."
        )
    }
]


def normalize_text(text: str) -> str:
    """
    Standardize Indonesian text: trim whitespace, normalize quotes and dashes.
    """
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('“', '"').replace('”', '"').replace("’", "'").replace("‘", "'")
    return text.strip()


def format_story_sample(item: Dict[str, Any]) -> str:
    """
    Format a story data item into special tokenized sequence for GPT-2 training:
    <|startoftext|> [GENRE: {genre}] [USIA: {age} tahun] [TEMA: {theme}] <|story|> {story} <|endoftext|>
    """
    genre = item.get("genre", "Umum")
    age = item.get("age", 7)
    theme = item.get("theme", "Petualangan")
    story = normalize_text(item.get("story", ""))
    
    formatted = (
        f"{SPECIAL_START} "
        f"[GENRE: {genre}] [USIA: {age} tahun] [TEMA: {theme}] "
        f"{SPECIAL_SEP} "
        f"{story} "
        f"{SPECIAL_END}"
    )
    return formatted


def prepare_and_save_dataset(
    output_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    val_size: float = 0.2,
    seed: int = 42
) -> Tuple[str, str]:
    """
    Create raw and processed datasets, split train/val, and save to JSON.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # Save raw stories
    raw_file = os.path.join(raw_dir, "indonesian_children_stories.json")
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(RAW_STORIES_COLLECTION, f, ensure_ascii=False, indent=2)

    formatted_samples = []
    for item in RAW_STORIES_COLLECTION:
        formatted_samples.append({
            "title": item["title"],
            "genre": item["genre"],
            "age": item["age"],
            "theme": item["theme"],
            "raw_story": item["story"],
            "formatted_text": format_story_sample(item)
        })

    train_data, val_data = train_test_split(
        formatted_samples, test_size=val_size, random_state=seed
    )

    train_file = os.path.join(output_dir, "train.json")
    val_file = os.path.join(output_dir, "val.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Dataset successfully prepared:")
    print(f" - Raw stories: {len(RAW_STORIES_COLLECTION)} items -> {raw_file}")
    print(f" - Train set: {len(train_data)} items -> {train_file}")
    print(f" - Val set:   {len(val_data)} items -> {val_file}")

    return train_file, val_file


if __name__ == "__main__":
    prepare_and_save_dataset()


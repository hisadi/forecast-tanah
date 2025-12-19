import io, os, sys, json, time
from pathlib import Path
from math import radians, sin, cos, asin, sqrt

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
    JOBLIB_OK = True
except Exception:
    JOBLIB_OK = False
    import pickle  # type: ignore

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns 
    MPL_OK = True
except Exception:
    MPL_OK = False

# ---------- PAGE CONFIG & CUSTOM CSS ----------
st.set_page_config(
    page_title="Prediksi Nilai Tanah • End User", 
    page_icon="🧭", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    /* Global Styling */
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #2c3e50; }
    
    /* Custom Card Container */
    .css-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .css-card:hover { transform: translateY(-2px); box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1); }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff; border: 1px solid #eee; padding: 15px;
        border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center;
    }
    
    /* Button Styling */
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; height: 3em; transition: all 0.3s ease; }
    
    /* Info Box Styling (Tips & Catatan) */
    .info-box-blue {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        color: #0d47a1;
        font-size: 14px;
        margin-bottom: 15px;
        line-height: 1.5;
    }
    .info-box-yellow {
        background-color: #fff9c4;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #fbc02d;
        color: #f57f17;
        font-size: 13px;
        margin-bottom: 15px;
        line-height: 1.5;
    }
    .sidebar-text { font-size: 14px; color: #555; }
</style>
""", unsafe_allow_html=True)

# ---------- path & import helpers ----------
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "pages" else THIS_FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def fmt_rp(x) -> str:
    try:
        if np.isnan(x): return "-"
        return f"Rp {float(x):,.0f}"
    except Exception:
        return str(x)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# ---------- CBD points (DKI) ----------
CBD_POINTS_JAKARTA = [
    {"cbd": "Bundaran HI / Thamrin",   "lat": -6.1930, "lon": 106.8220, "radius_km": 2.5},
    {"cbd": "Sudirman - Dukuh Atas",   "lat": -6.2000, "lon": 106.8200, "radius_km": 2.2},
    {"cbd": "Setiabudi",               "lat": -6.2080, "lon": 106.8240, "radius_km": 1.8},
    {"cbd": "Karet / Bendungan Hilir", "lat": -6.2100, "lon": 106.8090, "radius_km": 1.7},
    {"cbd": "Menteng - Cikini",        "lat": -6.1970, "lon": 106.8410, "radius_km": 2.0},
    {"cbd": "Monas - Gambir",          "lat": -6.1754, "lon": 106.8272, "radius_km": 2.0},
    {"cbd": "Tanah Abang",             "lat": -6.1940, "lon": 106.8080, "radius_km": 2.0},
    {"cbd": "Harmoni - Gajah Mada",    "lat": -6.1660, "lon": 106.8240, "radius_km": 1.8},
    {"cbd": "Senen - Salemba",         "lat": -6.1960, "lon": 106.8490, "radius_km": 2.2},
    {"cbd": "SCBD",                    "lat": -6.2251, "lon": 106.8096, "radius_km": 2.0},
    {"cbd": "Kuningan - Rasuna Said",  "lat": -6.2219, "lon": 106.8321, "radius_km": 2.2},
    {"cbd": "Mega Kuningan",           "lat": -6.2272, "lon": 106.8269, "radius_km": 1.8},
    {"cbd": "Gatot Subroto",           "lat": -6.2250, "lon": 106.8200, "radius_km": 2.0},
    {"cbd": "Senayan",                 "lat": -6.2277, "lon": 106.8070, "radius_km": 2.2},
    {"cbd": "Blok M",                  "lat": -6.2440, "lon": 106.8000, "radius_km": 2.0},
    {"cbd": "Senopati - Suryo",        "lat": -6.2360, "lon": 106.8130, "radius_km": 1.5},
    {"cbd": "Dharmawangsa",            "lat": -6.2480, "lon": 106.8020, "radius_km": 1.5},
    {"cbd": "Gandaria",                "lat": -6.2430, "lon": 106.7890, "radius_km": 1.7},
    {"cbd": "Kemang",                  "lat": -6.2600, "lon": 106.8150, "radius_km": 2.0},
    {"cbd": "Antasari",                "lat": -6.2670, "lon": 106.8110, "radius_km": 1.5},
    {"cbd": "Ampera",                  "lat": -6.2810, "lon": 106.8210, "radius_km": 1.4},
    {"cbd": "TB Simatupang",           "lat": -6.3000, "lon": 106.8200, "radius_km": 3.5},
    {"cbd": "Cilandak",                "lat": -6.2920, "lon": 106.8130, "radius_km": 1.5},
    {"cbd": "Fatmawati",               "lat": -6.2920, "lon": 106.8030, "radius_km": 1.7},
    {"cbd": "Pondok Indah",            "lat": -6.2770, "lon": 106.7840, "radius_km": 2.0},
    {"cbd": "Lebak Bulus",             "lat": -6.3000, "lon": 106.7730, "radius_km": 2.0},
    {"cbd": "Pejaten",                 "lat": -6.2830, "lon": 106.8400, "radius_km": 1.7},
    {"cbd": "Pancoran",                "lat": -6.2440, "lon": 106.8470, "radius_km": 1.8},
    {"cbd": "Tebet",                   "lat": -6.2360, "lon": 106.8580, "radius_km": 1.8},
    {"cbd": "Kalibata",                "lat": -6.2570, "lon": 106.8490, "radius_km": 1.6},
    {"cbd": "Pasar Minggu",            "lat": -6.2910, "lon": 106.8410, "radius_km": 2.0},
    {"cbd": "Slipi - Palmerah",        "lat": -6.2010, "lon": 106.7960, "radius_km": 2.0},
    {"cbd": "Tomang",                  "lat": -6.1780, "lon": 106.8020, "radius_km": 1.8},
    {"cbd": "Grogol - Tj. Duren",      "lat": -6.1590, "lon": 106.7900, "radius_km": 2.3},
    {"cbd": "Kebon Jeruk - Kedoya",    "lat": -6.2000, "lon": 106.7700, "radius_km": 2.5},
    {"cbd": "Puri Indah - Kembangan",  "lat": -6.1870, "lon": 106.7360, "radius_km": 2.8},
    {"cbd": "Daan Mogot",              "lat": -6.1660, "lon": 106.7350, "radius_km": 2.4},
    {"cbd": "Pluit",                   "lat": -6.1140, "lon": 106.7940, "radius_km": 2.5},
    {"cbd": "Muara Karang",            "lat": -6.1200, "lon": 106.7830, "radius_km": 2.0},
    {"cbd": "PIK (Pantai Indah Kapuk)","lat": -6.1150, "lon": 106.7420, "radius_km": 3.2},
    {"cbd": "Sunter",                  "lat": -6.1450, "lon": 106.8650, "radius_km": 2.2},
    {"cbd": "Ancol",                   "lat": -6.1270, "lon": 106.8420, "radius_km": 2.5},
    {"cbd": "Kota Tua",                "lat": -6.1350, "lon": 106.8130, "radius_km": 2.2},
    {"cbd": "Glodok - Mangga Besar",   "lat": -6.1490, "lon": 106.8160, "radius_km": 2.0},
    {"cbd": "Mangga Dua",              "lat": -6.1410, "lon": 106.8280, "radius_km": 1.8},
    {"cbd": "Kemayoran",               "lat": -6.1570, "lon": 106.8570, "radius_km": 2.5},
    {"cbd": "Kelapa Gading",           "lat": -6.1590, "lon": 106.9080, "radius_km": 3.0},
    {"cbd": "MOI / La Piazza",         "lat": -6.1560, "lon": 106.9020, "radius_km": 1.6},
    {"cbd": "Cempaka Putih",           "lat": -6.1770, "lon": 106.8700, "radius_km": 1.8},
    {"cbd": "Rawamangun",              "lat": -6.1940, "lon": 106.8900, "radius_km": 2.0},
    {"cbd": "Matraman",                "lat": -6.2080, "lon": 106.8640, "radius_km": 1.8},
    {"cbd": "Otista",                  "lat": -6.2470, "lon": 106.8610, "radius_km": 1.8},
    {"cbd": "Jatinegara",              "lat": -6.2210, "lon": 106.8820, "radius_km": 2.0},
    {"cbd": "Pulogadung",              "lat": -6.1930, "lon": 106.9020, "radius_km": 2.4},
    {"cbd": "Kramat Jati",             "lat": -6.2650, "lon": 106.8630, "radius_km": 2.0},
]

def pick_cbd_jakarta(lat, lon):
    best = ("Non-CBD/Other", float("inf"), None)
    for r in CBD_POINTS_JAKARTA:
        d = haversine_km(lat, lon, r["lat"], r["lon"])
        if d < best[1]:
            best = (r["cbd"], d, r)
    if best[2] and best[1] <= best[2]["radius_km"]:
        return best[0], round(best[1], 3)
    return "Non-CBD/Other", round(best[1], 3)

MANUAL_LABEL = "Manual… (ketik sendiri)"
ADDR_TREE = {
    "DKI Jakarta": {
        "Jakarta Pusat": {
            "Gambir": ["Cideng","Duri Pulo","Gambir","Kebon Kelapa","Petojo Selatan","Petojo Utara"],
            "Menteng": ["Cikini","Gondangdia","Menteng","Pegangsaan"],
            "Tanah Abang": ["Bendungan Hilir","Gelora","Karet Tengsin","Kebon Kacang","Kebon Melati","Petamburan"],
            "Senen": ["Bungur","Kenari","Kramat","Paseban","Senen"],
            "Johar Baru": ["Galur","Johar Baru","Kampung Rawa","Tanah Tinggi"],
            "Cempaka Putih": ["Cempaka Putih Barat","Cempaka Putih Timur","Rawasari"],
            "Kemayoran": ["Cempaka Baru","Gunung Sahari Selatan","Harapan Mulia","Kebon Kosong","Kemayoran","Serdang","Sumur Batu","Utan Panjang"],
            "Sawah Besar": ["Gunung Sahari Utara","Karang Anyar","Kartini","Mangga Dua Selatan"],
        },
        "Jakarta Selatan": {
            "Kebayoran Baru": ["Senayan","Selong","Rawa Barat","Gunung","Pulo","Melawai","Kramat Pela"],
            "Kebayoran Lama": ["Cipulir","Grogol Selatan","Grogol Utara","Kebayoran Lama Selatan","Kebayoran Lama Utara","Pondok Pinang"],
            "Cilandak": ["Cilandak Barat","Cipete Selatan","Gandaria Selatan","Lebak Bulus","Pondok Labu"],
            "Pasar Minggu": ["Cilandak Timur","Jati Padang","Kebagusan","Pejaten Barat","Pejaten Timur","Ragunan"],
            "Jagakarsa": ["Cipedak","Jagakarsa","Lenteng Agung","Srengseng Sawah","Tanjung Barat"],
            "Mampang Prapatan": ["Bangka","Kuningan Barat","Mampang Prapatan","Pela Mampang","Tegal Parang"],
            "Pancoran": ["Cikoko","Duren Tiga","Kalibata","Pancoran","Pengadegan","Rawa Jati"],
            "Setiabudi": ["Guntur","Karet","Karet Kuningan","Karet Semanggi","Kuningan Timur","Menteng Atas","Pasar Manggis"],
            "Tebet": ["Bukit Duri","Kebon Baru","Manggarai","Manggarai Selatan","Menteng Dalam","Tebet Barat","Tebet Timur"],
        },
        "Jakarta Barat": {
            "Grogol Petamburan": ["Grogol","Tanjung Duren Utara","Tanjung Duren Selatan","Tomang","Wijaya Kusuma"],
            "Tambora": ["Angke","Duri Selatan","Duri Utara","Jembatan Besi","Jembatan Lima","Kalianyar","Krendang","Pekojan","Roa Malaka","Tambora","Tanah Sereal"],
            "Cengkareng": ["Cengkareng Barat","Cengkareng Timur","Duri Kosambi","Kapuk","Kedaung Kali Angke","Rawa Buaya"],
            "Kalideres": ["Kalideres","Kamal","Pegadungan","Semanan","Tegal Alur"],
            "Kebon Jeruk": ["Duri Kepa","Kebon Jeruk","Kelapa Dua","Kedoya Selatan","Kedoya Utara","Sukabumi Selatan","Sukabumi Utara"],
            "Kembangan": ["Joglo","Kembangan Selatan","Kembangan Utara","Meruya Selatan","Meruya Utara","Srengseng"],
            "Palmerah": ["Palmerah","Kota Bambu Selatan","Kota Bambu Utara","Slipi","Kemanggisan"],
        },
        "Jakarta Timur": {
            "Matraman": ["Kayu Manis","Kebon Manggis","Pal Meriam","Pisangan Baru","Utan Kayu Selatan","Utan Kayu Utara"],
            "Pulo Gadung": ["Jati","Kayu Putih","Pisangan Timur","Pulo Gadung","Rawamangun"],
            "Kramat Jati": ["Balekambang","Batu Ampar","Cawang","Cililitan","Dukuh","Kramat Jati","Tengah"],
            "Jatinegara": ["Bali Mester","Bidara Cina","Cipinang Cempedak","Cipinang Besar Selatan","Cipinang Besar Utara","Cipinang Muara","Rawa Bunga"],
            "Duren Sawit": ["Duren Sawit","Malaka Jaya","Malaka Sari","Pondok Bambu","Pondok Kelapa","Pondok Kopi"],
            "Cakung": ["Cakung Barat","Cakung Timur","Penggilingan","Pulogebang","Rawa Terate","Ujung Menteng"],
            "Ciracas": ["Cibubur","Ciracas","Kelapa Dua Wetan","Rambutan","Susukan"],
            "Pasar Rebo": ["Baru","Cijantung","Gedong","Kalisari","Pekayon"],
            "Makasar": ["Cipinang Melayu","Halim Perdanakusuma","Kebon Pala","Makasar","Pinang Ranti"],
        },
        "Jakarta Utara": {
            "Penjaringan": ["Kapuk Muara","Kamal Muara","Muara Angke","Pejagalan","Pluit"],
            "Pademangan": ["Ancol","Pademangan Barat","Pademangan Timur"],
            "Tanjung Priok": ["Kebon Bawang","Papanggo","Sunter Agung","Sunter Jaya","Tanjung Priok","Warakas"],
            "Koja": ["Koja","Lagoa","Rawa Badak Selatan","Rawa Badak Utara","Tugu Selatan","Tugu Utara"],
            "Kelapa Gading": ["Kelapa Gading Barat","Kelapa Gading Timur","Pegangsaan Dua"],
            "Cilincing": ["Cilincing","Kalibaru","Marunda","Rorotan","Semper Barat","Semper Timur","Sukapura"],
        },
        "Kepulauan Seribu": {
            "Kepulauan Seribu Selatan": ["Pulau Pari","Pulau Tidung","Pulau Untung Jawa"],
            "Kepulauan Seribu Utara": ["Pulau Kelapa","Pulau Harapan","Pulau Panggang","Pulau Pramuka"],
        },
    },
    "Banten": {
        "Kota Tangerang": {"Batuceper": [], "Benda": [], "Cibodas": [], "Ciledug": [], "Cipondoh": [], "Jatiuwung": [],
                          "Karang Tengah": [], "Karawaci": [], "Larangan": [], "Neglasari": [], "Periuk": [], "Pinang": [], "Tangerang": []},
        "Kota Tangerang Selatan": {"Serpong": [], "Serpong Utara": [], "Pamulang": [], "Pondok Aren": [], "Ciputat": [], "Ciputat Timur": [], "Setu": []},
        "Kota Serang": {"Kasemen": [], "Curug": [], "Cipocok Jaya": [], "Serang": [], "Taktakan": [], "Walantaka": []},
        "Kota Cilegon": {"Cibeber": [], "Cilegon": [], "Citangkil": [], "Ciwandan": [], "Grogol": [], "Jombang": [], "Purwakarta": [], "Pulomerak": []},
        "Kabupaten Tangerang": {"Cikupa": [], "Curug": [], "Kelapa Dua": [], "Legok": [], "Pagedangan": [], "Panongan": [], "Pasar Kemis": [],
                                "Sindang Jaya": [], "Tigaraksa": [], "Balaraja": [], "Cisauk": [], "Sukamulya": []},
        "Kabupaten Serang": {"Anyar": [], "Cikande": [], "Ciruas": [], "Kragilan": [], "Kramatwatu": [], "MANCAK": [], "Waringinkurung": []},
        "Kabupaten Pandeglang": {"Pandeglang": [], "Labuan": [], "Menes": [], "Panimbang": [], "Carita": [], "Saketi": []},
        "Kabupaten Lebak": {"Rangkasbitung": [], "Cibadak": [], "Maja": [], "Warunggunung": [], "Bayah": [], "Cikulur": []},
    },
    "Jawa Barat": {
        "Kota Bandung": {"Coblong": [], "Sukajadi": [], "Cidadap": [], "Bandung Wetan": [], "Sumur Bandung": [],
                         "Lengkong": [], "Bojongloa Kaler": [], "Bojongloa Kidul": [], "Antapani": [], "Kiaracondong": []},
        "Kota Bekasi": {"Bekasi Barat": [], "Bekasi Timur": [], "Bekasi Utara": [], "Bekasi Selatan": [], "Medan Satria": [], "Pondok Gede": [], "Jatisampurna": []},
        "Kota Depok": {"Beji": [], "Cimanggis": [], "Pancoran Mas": [], "Sukmajaya": [], "Tapos": [], "Limo": [], "Cinere": []},
        "Kota Bogor": {"Bogor Tengah": [], "Bogor Timur": [], "Bogor Barat": [], "Bogor Utara": [], "Bogor Selatan": [], "Tanah Sereal": []},
        "Kota Cimahi": {"Cimahi Utara": [], "Cimahi Tengah": [], "Cimahi Selatan": []},
        "Kota Cirebon": {"Kejaksan": [], "Lemahwungkuk": [], "Harjamukti": [], "Pekalipan": [], "Kesambi": []},
        "Kota Sukabumi": {"Cikole": [], "Citamiang": [], "Gunungpuyuh": [], "Lembursitu": [], "Warudoyong": []},
        "Kota Tasikmalaya": {"Cihideung": [], "Kawalu": [], "Mangkubumi": [], "Tamansari": [], "Cipedes": []},
        "Kabupaten Bogor": {"Cibinong": [], "Bojong Gede": [], "Gunung Putri": [], "Citeureup": [], "Ciawi": [], "Parung": []},
        "Kabupaten Bekasi": {"Cikarang Barat": [], "Cikarang Timur": [], "Cikarang Utara": [], "Tambun Selatan": [], "Tambun Utara": [], "Setu": []},
        "Kabupaten Bandung": {"Soreang": [], "Baleendah": [], "Bojongsoang": [], "Margahayu": [], "Cileunyi": [], "Dayeuhkolot": []},
        "Kabupaten Bandung Barat": {"Ngamprah": [], "Padalarang": [], "Lembang": [], "Batujajar": [], "Cikalongwetan": []},
        "Kabupaten Karawang": {"Karawang Barat": [], "Karawang Timur": [], "Telukjambe Timur": [], "Telukjambe Barat": [], "Cikampek": []},
        "Kabupaten Purwakarta": {"Purwakarta": [], "Jatiluhur": [], "Babakancikao": []},
        "Kabupaten Subang": {"Subang": [], "Kalijati": [], "Ciasem": []},
        "Kabupaten Indramayu": {"Indramayu": [], "Jatibarang": [], "Haurgeulis": []},
        "Kabupaten Cirebon": {"Sumber": [], "Weru": [], "Arjawinangun": []},
        "Kabupaten Garut": {"Garut Kota": [], "Tarogong Kidul": [], "Tarogong Kaler": [], "Cibatu": []},
        "Kabupaten Sukabumi": {"Cisaat": [], "Cibadak": [], "Pelabuhanratu": [], "Cicurug": []},
        "Kabupaten Sumedang": {"Sumedang Utara": [], "Sumedang Selatan": [], "Jatinangor": [], "Tanjungsari": []},
        "Kabupaten Tasikmalaya": {"Singaparna": [], "Manonjaya": [], "Rajapolah": []},
        "Kota Banjar": {"Banjar": [], "Langensari": [], "Pataruman": []},
    },
}

# ---------- [CRITICAL] DATA CLEANING FUNCTION ----------
def clean_and_standardize_data(df):
    elevasi_map = {
        "sama dengan jalan": "Sama Dengan Jalan", "Sama dengan jalan": "Sama Dengan Jalan", "Sama Dengan Jalan": "Sama Dengan Jalan",
        "lebih tinggi dari jalan": "Lebih Tinggi", "lebih tinggi": "Lebih Tinggi", "Lebih tinggi": "Lebih Tinggi", "Lebih Tinggi": "Lebih Tinggi",
        "lebih rendah dari jalan": "Lebih Rendah", "lebih rendah": "Lebih Rendah", "Lebih rendah": "Lebih Rendah", "Lebih Rendah": "Lebih Rendah",
    }
    kontur_map = {
        "datar": "Datar", "Datar": "Datar", "1 datar": "Datar", "2 datar": "Datar", "datar dan butuh uruk": "Datar",
        "bergelombang": "Bergelombang", "Bergelombang": "Bergelombang",
        "miring": "Miring", "Miring": "Miring", "Miring-Mendaki": "Miring",
        "terasering": "Terasering", "Terasering": "Terasering"
    }
    
    if "elavasi" in df.columns:
        df["elavasi"] = df["elavasi"].astype(str).str.strip().map(elevasi_map).fillna("Datar")
    if "kontur" in df.columns:
        df["kontur"] = df["kontur"].astype(str).str.strip().map(kontur_map).fillna("Rata")
        
    cols_title = ["kondisi_jalan", "kontruksi_jalan", "pemanfaatan_sekitar", "dokumen_kepemilikan"]
    for col in cols_title:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df

# ---------- load default model/config ----------
def _coerce_to_predictor(obj):
    try:
        if hasattr(obj, "predict") and callable(getattr(obj, "predict")): return obj
        if isinstance(obj, dict):
            for k in ("model","pipeline","pipe","estimator"):
                if k in obj and hasattr(obj[k], "predict"): return obj[k]
        if isinstance(obj, (list, tuple)):
            for v in obj:
                p = _coerce_to_predictor(v)
                if p is not None: return p
    except: pass
    return None

@st.cache_resource(show_spinner=False)
def load_default_model_and_config():
    models_dir = ROOT / "models"
    cfg = None; model = None
    if models_dir.exists():
        cfg_path = models_dir / "config_latest.json"
        if cfg_path.exists():
            try: cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except: cfg = None
        for fname in ("model_bundle_latest.pkl", "model_latest.pkl"):
            f = models_dir / fname
            if f.exists():
                try:
                    obj = joblib.load(f) if JOBLIB_OK else pickle.load(open(f, "rb"))
                    p = _coerce_to_predictor(obj)
                    if p is not None: model = p; break
                except: continue
    return model, cfg

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2642/2642502.png", width=50) 
    st.title("Konfigurasi Sistem")
    st.markdown("---")
    
    st.subheader("🤖 Status Model")
    model_obj = st.session_state.get("trained_model")
    feature_cfg = st.session_state.get("feature_cfg")

    use_default = st.toggle("Pakai Model Default", value=True, help="Otomatis memuat dari folder ./models")
    
    if use_default:
        if model_obj is None:
            with st.spinner("Memuat model default..."):
                m, c = load_default_model_and_config()
                if m is not None:
                    model_obj = m
                    st.session_state["trained_model"] = m
                    st.success("✅ Model Default Aktif")
                if c is not None:
                    feature_cfg = c
                    st.session_state["feature_cfg"] = c
    else:
        st.info("Mode Manual Aktif")

    if not use_default or model_obj is None:
        up = st.file_uploader("Upload Pipeline (.pkl/.joblib)", type=["pkl","joblib","bin"])
        if up is not None:
            try:
                raw = joblib.load(up) if JOBLIB_OK else pickle.load(up)
                pred = _coerce_to_predictor(raw)
                if pred:
                    model_obj = pred
                    st.session_state["trained_model"] = model_obj
                    st.success("✅ Model Terupload")
                else:
                    st.error("File tidak valid.")
            except Exception as e:
                st.error(f"Error: {e}")

        cfg_up = st.file_uploader("Upload Config (.json)", type=["json"])
        if cfg_up:
            try:
                feature_cfg = json.loads(cfg_up.read().decode("utf-8"))
                st.session_state["feature_cfg"] = feature_cfg
                st.success("✅ Config Dimuat")
            except:
                st.error("Config error")

    # --- [NEW] BAGIAN CARA PENGGUNAAN DI SIDEBAR ---
    st.markdown("---")
    st.subheader("💡 Cara Penggunaan")
    st.markdown("""
    <div class="sidebar-text">
    1. <b>Isi Parameter:</b> Lengkapi data lokasi, fisik tanah, dan legalitas di formulir utama.
    2. <b>Cek Peta/Lokasi:</b> Pastikan Lat/Lon sesuai untuk akurasi jarak ke CBD.
    3. <b>Klik Prediksi:</b> Tekan tombol biru di bawah formulir.
    4. <b>Analisis:</b> Lihat hasil valuasi dan faktor yang mempengaruhinya.
    </div>
    """, unsafe_allow_html=True)
    # -----------------------------------------------

    st.markdown("---")
    st.caption("Powered By Binus Skripsi © 2025/2026")

# ==============================================================================
# MAIN PAGE
# ==============================================================================

st.markdown("""
<div style="padding: 20px; background: linear-gradient(90deg, #1a2980 0%, #26d0ce 100%); border-radius: 15px; color: white; margin-bottom: 30px;">
    <h1 style="color: white; margin:0;">🧭 Prediksi Nilai Tanah</h1>
    <p style="margin:0; opacity: 0.9;">Estimasi harga pasar dengan <b>Standardisasi Data Otomatis</b>.</p>
</div>
""", unsafe_allow_html=True)

if model_obj is None:
    st.warning("⚠️ Model belum dimuat. Silakan cek sidebar.")
    st.stop()

# --- LAYOUT BARU: KIRI INPUT (70%), KANAN TIPS (30%) ---
col_left, col_right = st.columns([2.5, 1])

# --- KOLOM KIRI: SEMUA INPUT FORM ---
with col_left:
    st.subheader("📝 Input Parameter Tanah")
    
    # GROUP 1: LOKASI
    with st.container():
        st.markdown("---")
        st.markdown("### 📍 Lokasi & Geografis")
        
        c1, c2 = st.columns(2)
        with c1: lat = st.number_input("Latitude", value=-6.200000, format="%.6f")
        with c2: lon = st.number_input("Longitude", value=106.816666, format="%.6f")

        cbd_auto, dist_auto = pick_cbd_jakarta(lat, lon)
        st.info(f"🎯 **CBD Terdekat (Auto):** {cbd_auto} ({dist_auto} km)", icon="🏢")

        st.markdown("---")
        provinsi_list = list(ADDR_TREE.keys()) + [MANUAL_LABEL]
        prov_choice = st.selectbox("Provinsi", provinsi_list, index=0)
        provinsi_val = st.text_input("✍️ Provinsi Manual", "") if prov_choice == MANUAL_LABEL else prov_choice

        kota_opts = [MANUAL_LABEL] if prov_choice == MANUAL_LABEL else list(ADDR_TREE.get(provinsi_val, {}).keys()) + [MANUAL_LABEL]
        kota_choice = st.selectbox("Kota/Kabupaten", kota_opts)
        kota_kabupaten = st.text_input("✍️ Kota Manual", "") if kota_choice == MANUAL_LABEL else kota_choice
        
        kec_opts = [MANUAL_LABEL]
        if kota_choice != MANUAL_LABEL and prov_choice != MANUAL_LABEL:
             kec_opts = list(ADDR_TREE.get(provinsi_val, {}).get(kota_kabupaten, {}).keys()) + [MANUAL_LABEL]
        
        kc1, kc2 = st.columns(2)
        with kc1:
            kec_choice = st.selectbox("Kecamatan", kec_opts)
            kecamatan = st.text_input("✍️ Kec. Manual", "") if kec_choice == MANUAL_LABEL else kec_choice
        with kc2:
            kel_list = []
            if kec_choice not in (MANUAL_LABEL, ""):
                kel_list = ADDR_TREE.get(provinsi_val, {}).get(kota_kabupaten, {}).get(kecamatan, [])
            kel_opts = (kel_list if kel_list else []) + [MANUAL_LABEL]
            kel_choice = st.selectbox("Kelurahan", kel_opts)
            kelurahan = st.text_input("✍️ Kel. Manual", "") if kel_choice == MANUAL_LABEL else kel_choice
            
        st.markdown('</div>', unsafe_allow_html=True)

    # GROUP 2: SPESIFIKASI LAHAN
    with st.container():
        st.markdown("---")
        st.markdown("### 📐 Spesifikasi & Kondisi Lahan")
        
        l1, l2 = st.columns(2)
        with l1:
            luas = st.number_input("Luas Tanah (m²)", value=100.0, min_value=1.0, step=10.0)
            jarak_ke_jalan = st.number_input("Jarak Ke Jalan (m)", value=6.0, step=0.5)
        with l2:
            kontur = st.selectbox("Kontur Tanah", ["Datar", "Bergelombang", "Miring", "Terasering"])
            elavasi = st.selectbox("Elevasi", ["Sama Dengan Jalan", "Lebih Rendah", "Lebih Tinggi"])
            
        k1, k2 = st.columns(2)
        with k1: kontruksi_jalan = st.selectbox("Konstruksi Jalan", ["Aspal","Beton","Paving","Tanah"])
        with k2: kondisi_jalan = st.selectbox("Kondisi Jalan", ["Baik","Sedang","Buruk","Rusak"])
        st.markdown('</div>', unsafe_allow_html=True)

    # GROUP 3: LEGALITAS
    with st.container():
        st.markdown("---")
        st.markdown("### ⚖️ Legalitas & Lingkungan")
        dokumen_kepemilikan = st.selectbox("Dokumen", ["SHM","HGB","HPL","Girik/AJB","Lainnya"])
        pemanfaatan_sekitar = st.selectbox("Pemanfaatan Sekitar", ["Perumahan","Komersial","Campuran","Industri","Lahan Kosong","Pertanian"])
        jenis_transaksi = st.selectbox("Jenis Transaksi", ["Jual","Sewa","Lelang"])
        sumber_data = st.selectbox("Sumber Data", ["Iklan","Survey Lapangan","Agen/PPAT","Lainnya"])
        
        st.markdown("---")
        st.markdown("**Override CBD (Opsional)**")
        cbd_options = ["Non-CBD/Other"] + [r["cbd"] for r in CBD_POINTS_JAKARTA]
        def_idx = cbd_options.index(cbd_auto) if cbd_auto in cbd_options else 0
        nama_cbd = st.selectbox("Nama CBD Reference", options=cbd_options, index=def_idx)
        jarak_cbd = st.number_input("Jarak ke CBD (km)", value=float(dist_auto), step=0.1, format="%.3f")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # GROUP 4: SETTING
    with st.container():
        st.markdown('<div class="css-card" style="background-color:#f0f2f6;">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Setting Analisis")
        ignore_latlon = st.checkbox("Abaikan Lat/Lon di analisis fitur", value=True)
        SENS_PCT = st.slider("Sensitivitas Numerik (±%)", 1, 20, 5, 1) / 100.0
        st.markdown('</div>', unsafe_allow_html=True)

# --- KOLOM KANAN: TIPS & CATATAN (STICKY) ---
with col_right:
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    
    st.subheader("🔍 Tips Prediksi")
    st.markdown("""
    <div class="info-box-blue">
        <b>Tips untuk hasil akurat:</b>
        <ul style="margin-top: 5px; padding-left: 20px;">
            <li><b>Luas Tanah:</b> Pastikan input luas sesuai sertifikat.</li>
            <li><b>Elevasi:</b> Tanah yang "Lebih Rendah" dari jalan biasanya memiliki valuasi lebih rendah karena risiko banjir.</li>
            <li><b>Legalitas:</b> Status SHM menaikkan nilai tanah dibanding Girik/AJB.</li>
            <li><b>Lokasi:</b> Jarak ke CBD (Pusat Kota) sangat mempengaruhi harga secara eksponensial.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("⚠️ Catatan Penting")
    st.markdown("""
    <div class="info-box-yellow">
        <b>Disclaimer:</b>
        <p>Prediksi ini menggunakan algoritma Machine Learning berdasarkan data historis pasar.</p>
        <p>Harga aktual dapat berbeda tergantung:</p>
        <ul style="padding-left: 20px;">
            <li>Kondisi ekonomi terkini</li>
            <li>Bentuk tanah (ngantong/kotak)</li>
            <li>Negosiasi penjual-pembeli</li>
            <li>Faktor estetika & view</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.image("https://img.freepik.com/free-photo/delimitation-land-plots_23-2150170946.jpg", caption="Prediksi Tanah AI", use_container_width=True)

# PREPARE DATA
DEFAULT_REQUIRED_COLS = [
    "nama_cbd","sumber_data","elavasi","jarak_ke_jalan","kontur",
    "kontruksi_jalan","kota_kabupaten","kondisi_jalan","jenis_transaksi",
    "luas","dokumen_kepemilikan","jarak_cbd","pemanfaatan_sekitar",
    "latitude","longitude","provinsi"
]
required_cols = feature_cfg.get("required_cols", DEFAULT_REQUIRED_COLS) if isinstance(feature_cfg, dict) else DEFAULT_REQUIRED_COLS

row = {
    "nama_cbd": nama_cbd, "sumber_data": sumber_data, "elavasi": elavasi,
    "jarak_ke_jalan": float(jarak_ke_jalan), "kontur": kontur, "kontruksi_jalan": kontruksi_jalan,
    "kota_kabupaten": kota_kabupaten, "kondisi_jalan": kondisi_jalan, "jenis_transaksi": jenis_transaksi,
    "luas": float(luas), "dokumen_kepemilikan": dokumen_kepemilikan, "jarak_cbd": float(jarak_cbd),
    "pemanfaatan_sekitar": pemanfaatan_sekitar, "latitude": float(lat), "longitude": float(lon),
    "provinsi": provinsi_val, "_kecamatan": kecamatan, "_kelurahan": kelurahan,
}
X_pred = pd.DataFrame([[row.get(c, np.nan) for c in required_cols]], columns=required_cols)

# ==============================================================================
# PREDICTION ACTION
# ==============================================================================
st.markdown("###")
predict_btn = st.button("🚀 HITUNG PREDIKSI & ANALISIS", type="primary", use_container_width=True)

# Helper Functions
def _is_number(x): return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

def explain_numeric_local(model, X_row, pct, skip_cols=None):
    skip = set(skip_cols or [])
    base = float(model.predict(X_row)[0])
    recs = []
    for col in X_row.columns:
        if col in skip: continue
        v = X_row.iloc[0][col]
        if _is_number(v):
            delta = pct * (abs(v) if v != 0 else 1.0)
            up = v + delta; dn = v - delta
            Xu = X_row.copy(); Xd = X_row.copy()
            Xu.iloc[0, X_row.columns.get_loc(col)] = up
            Xd.iloc[0, X_row.columns.get_loc(col)] = dn
            try:
                pu = float(model.predict(Xu)[0]); pdn = float(model.predict(Xd)[0])
                grad = (pu - pdn) / (up - dn + 1e-12)
                abs_effect = max(abs(pu - base), abs(pdn - base))
                recs.append({
                    "feature": col, "type": "numeric", "current": v, "base_pred": base,
                    "effect_up": pu - base, "effect_down": pdn - base, "sensitivity": grad, "abs_effect": abs_effect
                })
            except: continue
    return pd.DataFrame(recs)

def explain_categorical_contrast(model, X_row, choices_map, skip_cols=None):
    skip = set(skip_cols or [])
    base = float(model.predict(X_row)[0])
    recs = []
    for col in X_row.columns:
        if col in skip: continue
        if col in choices_map:
            curr = X_row.iloc[0][col]
            alts = [a for a in choices_map[col] if a != curr]
            preds = []
            for alt in alts:
                Xtmp = X_row.copy()
                Xtmp.iloc[0, X_row.columns.get_loc(col)] = alt
                try: preds.append((alt, float(model.predict(Xtmp)[0])))
                except: continue
            if preds:
                avg_alt = float(np.mean([p for _, p in preds]))
                delta_vs_avg = base - avg_alt
                best_alt, best_pred = max(preds, key=lambda t: abs(t[1] - base))
                recs.append({
                    "feature": col, "type": "categorical", "current": curr, "base_pred": base,
                    "delta_vs_avg_alt": delta_vs_avg, "best_alt": best_alt, "effect_if_best_alt": best_pred - base,
                    "n_valid_alts": len(preds), "abs_effect": abs(delta_vs_avg)
                })
    return pd.DataFrame(recs)

# Update Choices UI dengan Kategori Standar
CAT_CHOICES_UI = {
    "elavasi": ["Sama Dengan Jalan", "Lebih Rendah", "Lebih Tinggi"],
    "kontur": ["Datar", "Bergelombang", "Miring", "Terasering"],
    "kontruksi_jalan": ["Aspal","Beton","Paving","Tanah"],
    "kondisi_jalan": ["Baik","Sedang","Buruk","Rusak"],
    "pemanfaatan_sekitar": ["Perumahan","Komersial","Campuran","Industri","Lahan Kosong","Pertanian"],
    "dokumen_kepemilikan": ["SHM","HGB","HPL","Girik/AJB","Lainnya"],
    "jenis_transaksi": ["Jual","Sewa","Lelang"],
    "sumber_data": ["Iklan","Survey Lapangan","Agen/PPAT","Lainnya"],
    "nama_cbd": ["Non-CBD/Other"] + [r["cbd"] for r in CBD_POINTS_JAKARTA],
    "provinsi": list(ADDR_TREE.keys()),
    "kota_kabupaten": list(ADDR_TREE.get(provinsi_val, {}).keys()),
}

if predict_btn:
    progress_text = "Sedang menganalisis..."
    my_bar = st.progress(0, text=progress_text)

    try:
        # --- [CRITICAL UPDATE] CLEAN DATA ---
        X_pred = clean_and_standardize_data(X_pred)
        # -------------------------------------

        # Simulation animation
        for percent_complete in range(0, 40, 10):
            time.sleep(0.05)
            my_bar.progress(percent_complete, text="Validasi input...")
        
        y_hat = float(model_obj.predict(X_pred)[0])
        rpm2 = (y_hat / float(luas)) if (luas and luas > 0) else np.nan
        
        for percent_complete in range(40, 80, 10):
            time.sleep(0.05)
            my_bar.progress(percent_complete, text="Menghitung kontribusi fitur...")

        # Analisis
        skip_cols = {"_kecamatan","_kelurahan"}
        if ignore_latlon: skip_cols |= {"latitude","longitude"}
        
        df_num = explain_numeric_local(model_obj, X_pred, pct=SENS_PCT, skip_cols=skip_cols)
        df_cat = explain_categorical_contrast(model_obj, X_pred, CAT_CHOICES_UI, skip_cols=skip_cols)
        
        my_bar.progress(100, text="Selesai!")
        time.sleep(0.2)
        my_bar.empty()
        st.toast("Prediksi Selesai!", icon="✅")

        # --- RESULT DISPLAY ---
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Estimasi Harga Total", value=fmt_rp(y_hat), delta="Output Model")
        with res_col2:
            st.metric(label="Harga per Meter Persegi", value=fmt_rp(rpm2), delta_color="off")

        # --- EXPLANATION TABS ---
        st.markdown("### 🔍 Analisis Faktor Penentu")
        
        parts = []
        if df_num is not None and not df_num.empty: parts.append(df_num)
        if df_cat is not None and not df_cat.empty: parts.append(df_cat)
        
        if parts:
            contrib = pd.concat(parts, ignore_index=True)
            contrib = contrib.sort_values("abs_effect", ascending=False).reset_index(drop=True)
            total_abs = contrib["abs_effect"].sum() if len(contrib) else 0.0
            contrib["abs_share"] = (contrib["abs_effect"] / total_abs) if total_abs > 0 else 0.0
            
            def _direction(row):
                if row["type"] == "numeric":
                    s = row.get("sensitivity", 0.0)
                    return "Positif (Menaikkan)" if s > 0 else "Negatif (Menurunkan)"
                else:
                    d = row.get("delta_vs_avg_alt", 0.0)
                    return "Di atas Rerata" if d > 0 else "Di bawah Rerata"
            
            contrib["Karakter"] = contrib.apply(_direction, axis=1)
            
            tab1, tab2, tab3 = st.tabs(["💡 Top Features", "📈 Detail Numerik", "📋 Detail Kategorik"])
            
            with tab1:
                st.caption("Faktor-faktor dengan dampak absolut terbesar terhadap harga.")
                top_features = contrib.head(10)[["feature", "abs_effect", "type", "Karakter", "abs_share"]]
                for _, r in top_features.iterrows():
                    pct = r['abs_share'] * 100
                    color = "#2ecc71" if "Positif" in r['Karakter'] or "atas" in r['Karakter'] else "#e74c3c"
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.9em; font-weight:bold;">
                            <span>{r['feature']} <span style="font-weight:normal; color:#666; font-size:0.8em;">({r['type']})</span></span>
                            <span>{fmt_rp(r['abs_effect'])}</span>
                        </div>
                        <div style="width:100%; background:#eee; height:8px; border-radius:4px; margin-top:2px;">
                            <div style="width:{min(pct*2, 100)}%; background:{color}; height:100%; border-radius:4px;"></div>
                        </div>
                        <div style="font-size:0.75em; color:#888; text-align:right;">Pengaruh: {pct:.1f}% • {r['Karakter']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with tab2: st.dataframe(df_num.style.format({"effect_up": "{:,.0f}", "effect_down": "{:,.0f}", "sensitivity": "{:.2f}"}), use_container_width=True)
            with tab3: st.dataframe(df_cat.style.format({"delta_vs_avg_alt": "{:,.0f}", "effect_if_best_alt": "{:,.0f}"}), use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")

# ==============================================================================
# BATCH PREDICTION SECTION
# ==============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("📂 Prediksi Batch (Upload File)", expanded=False):
    st.info("Upload file Excel/CSV dengan kolom fitur yang sama untuk prediksi massal.")
    batch_file = st.file_uploader("Upload Data", type=["csv","xlsx"], key="batch_file")
    
    if batch_file:
        if batch_file.name.lower().endswith(".csv"): df_in = pd.read_csv(batch_file)
        else: df_in = pd.read_excel(batch_file)
        
        st.write(f"Preview ({len(df_in)} baris):")
        st.dataframe(df_in.head(), use_container_width=True)
        
        if st.button("Proses Batch"):
            defaults_for_missing = {"sumber_data": "Iklan", "elavasi": "Datar", "kontur": "Rata", "kontruksi_jalan": "Aspal", "kondisi_jalan": "Baik", "jenis_transaksi": "Jual", "dokumen_kepemilikan": "SHM", "pemanfaatan_sekitar": "Perumahan", "luas": 100.0, "jarak_ke_jalan": 50.0, "provinsi": "DKI Jakarta", "nama_cbd": "Non-CBD/Other"}
            for col in defaults_for_missing:
                if col not in df_in.columns: df_in[col] = defaults_for_missing[col]
            
            try:
                # --- [CRITICAL UPDATE] CLEAN DATA ---
                # Bersihkan data Excel yang kotor sebelum masuk model
                df_in = clean_and_standardize_data(df_in)
                # -------------------------------------

                Xb = df_in[required_cols].copy()
                y_batch = model_obj.predict(Xb)
                df_in["Prediksi_Harga"] = y_batch
                
                st.success("Selesai!")
                st.dataframe(df_in[["Prediksi_Harga"] + required_cols[:3]].head(), use_container_width=True)
                
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    df_in.to_excel(w, index=False)
                st.download_button("⬇️ Download Hasil Excel", data=buf.getvalue(), file_name="hasil_prediksi.xlsx")
            except Exception as e:
                st.error(f"Error batch: {e}")
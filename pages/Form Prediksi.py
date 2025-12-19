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
CBD_POINTS_JAKARTA = [{'cbd': 'AEON Jakarta Garden City', 'lat': -6.155535, 'lon': 106.96006, 'radius_km': 3.96}, {'cbd': 'aeon jgc cakung', 'lat': -6.204614, 'lon': 106.959887, 'radius_km': 6.0}, {'cbd': 'AEON Mall', 'lat': -6.170278, 'lon': 106.95212, 'radius_km': 2.54}, {'cbd': 'Aeon Mall Jakarta Garden City', 'lat': -6.167722, 'lon': 106.963306, 'radius_km': 4.2}, {'cbd': 'Aeon Mall JGC', 'lat': -6.179085, 'lon': 106.955886, 'radius_km': 3.22}, {'cbd': 'AEON Mall Tanjung Barat', 'lat': -6.301133, 'lon': 106.839398, 'radius_km': 2.77}, {'cbd': 'AEON Mall Tj Barat', 'lat': -6.300507, 'lon': 106.828651, 'radius_km': 3.38}, {'cbd': 'AEON tanjung barat', 'lat': -6.312971, 'lon': 106.826237, 'radius_km': 6.0}, {'cbd': 'AEON TJ Barat', 'lat': -6.307096, 'lon': 106.835972, 'radius_km': 2.99}, {'cbd': 'amaris hotel senen', 'lat': -6.18226, 'lon': 106.84392, 'radius_km': 1.0}, {'cbd': 'Apartemen Puri Imperium', 'lat': -6.207227, 'lon': 106.830617, 'radius_km': 1.0}, {'cbd': 'Apartemen Puri Orchard', 'lat': -6.164583, 'lon': 106.727583, 'radius_km': 1.0}, {'cbd': 'Apartemen Puru Orchard', 'lat': -6.163698, 'lon': 106.727345, 'radius_km': 1.0}, {'cbd': 'Apartemen Somerset Berlian Jakarta', 'lat': -6.22318, 'lon': 106.781774, 'radius_km': 1.0}, {'cbd': 'Apartemen West Vista', 'lat': -6.161049, 'lon': 106.724582, 'radius_km': 1.0}, {'cbd': 'arandra residence', 'lat': -6.177716, 'lon': 106.871752, 'radius_km': 1.0}, {'cbd': 'Area Cibis Park', 'lat': -6.300319, 'lon': 106.831345, 'radius_km': 3.0}, {'cbd': 'Area ITC Fatmawati', 'lat': -6.262164, 'lon': 106.79698, 'radius_km': 1.0}, {'cbd': 'Ario Mall', 'lat': -6.190812, 'lon': 106.893276, 'radius_km': 0.75}, {'cbd': 'arion', 'lat': -6.206294, 'lon': 106.890322, 'radius_km': 2.0}, {'cbd': 'Arion Mall', 'lat': -6.195042, 'lon': 106.887893, 'radius_km': 1.78}, {'cbd': 'Arion XXI', 'lat': -6.204671, 'lon': 106.889993, 'radius_km': 2.0}, {'cbd': 'Astha District 8', 'lat': -6.234328, 'lon': 106.808193, 'radius_km': 1.25}, {'cbd': 'Atrium Mall Kelapa Gading', 'lat': -6.148829, 'lon': 106.9165, 'radius_km': 2.0}, {'cbd': 'Atrium Senen', 'lat': -6.180729, 'lon': 106.840162, 'radius_km': 4.15}, {'cbd': 'Bandara City Mall', 'lat': -6.10002, 'lon': 106.703973, 'radius_km': 4.35}, {'cbd': 'Bandara Halim Perdana Kusuma', 'lat': -6.32, 'lon': 106.85, 'radius_km': 6.0}, {'cbd': 'Bandara Internasional Halim Perdana Kusuma', 'lat': -6.274662, 'lon': 106.819054, 'radius_km': 6.0}, {'cbd': 'BANDARA SOEKARNO HAT', 'lat': -6.127875, 'lon': 106.743665, 'radius_km': 6.0}, {'cbd': 'bandara soetta', 'lat': -6.135114, 'lon': 106.671717, 'radius_km': 4.0}, {'cbd': 'BCA Sunter Danau', 'lat': -6.145612, 'lon': 106.859995, 'radius_km': 1.0}, {'cbd': 'beach city jakarta', 'lat': -6.121575, 'lon': 106.857104, 'radius_km': 1.8}, {'cbd': 'Becakayu', 'lat': -6.245729, 'lon': 106.911179, 'radius_km': 3.0}, {'cbd': 'Bekasi CBD', 'lat': -6.180074, 'lon': 106.898075, 'radius_km': 6.0}, {'cbd': 'Bella Terra Lifestyle Center', 'lat': -6.180109, 'lon': 106.898119, 'radius_km': 0.72}, {'cbd': 'Bellagio Boutique Mall', 'lat': -6.227887, 'lon': 106.821433, 'radius_km': 0.72}, {'cbd': 'Bintaro Plaza', 'lat': -6.271262, 'lon': 106.757087, 'radius_km': 2.37}, {'cbd': 'Bintaro Sektor 5A', 'lat': -6.258431, 'lon': 106.760174, 'radius_km': 3.82}, {'cbd': 'Bizpark Commercial Estate', 'lat': -6.182552, 'lon': 106.912274, 'radius_km': 2.1}, {'cbd': 'Blok M', 'lat': -6.246485, 'lon': 106.809906, 'radius_km': 2.47}, {'cbd': 'blok M plasa', 'lat': -6.252067, 'lon': 106.813366, 'radius_km': 4.02}, {'cbd': 'blok M square', 'lat': -6.247573, 'lon': 106.804717, 'radius_km': 2.8}, {'cbd': 'Buaran Plaza', 'lat': -6.23098, 'lon': 106.920222, 'radius_km': 2.98}, {'cbd': 'bundaran HI', 'lat': -6.191583, 'lon': 106.836444, 'radius_km': 2.5}, {'cbd': 'Business Park Kb. Jeruk', 'lat': -6.197434, 'lon': 106.765028, 'radius_km': 0.5}, {'cbd': 'Carrefour Cempaka Putih', 'lat': -6.172701, 'lon': 106.884405, 'radius_km': 0.95}, {'cbd': 'CBC CENGKARENG BUSSINESS CITY', 'lat': -6.098509, 'lon': 106.707115, 'radius_km': 2.0}, {'cbd': 'CBD Ciledug', 'lat': -6.219658, 'lon': 106.733437, 'radius_km': 2.8}, {'cbd': 'CBD Ciledug Family Mall', 'lat': -6.221412, 'lon': 106.724476, 'radius_km': 2.76}, {'cbd': 'CBD Mega Kuningan', 'lat': -6.220197, 'lon': 106.824413, 'radius_km': 2.26}, {'cbd': 'Central Cakung Business Park', 'lat': -6.128996, 'lon': 106.931999, 'radius_km': 3.0}, {'cbd': 'Central Park', 'lat': -6.177083, 'lon': 106.795419, 'radius_km': 1.42}, {'cbd': 'central park mall', 'lat': -6.192707, 'lon': 106.771435, 'radius_km': 5.57}, {'cbd': 'Cibis Tower', 'lat': -6.309737, 'lon': 106.810143, 'radius_km': 2.11}, {'cbd': 'Cibubur Juction', 'lat': -6.326576, 'lon': 106.886629, 'radius_km': 6.0}, {'cbd': 'cibubur junction', 'lat': -6.350262, 'lon': 106.893855, 'radius_km': 5.76}, {'cbd': 'Cibubur Plaza', 'lat': -6.383381, 'lon': 106.913653, 'radius_km': 1.1}, {'cbd': 'Cibubur Square', 'lat': -6.339977, 'lon': 106.8975, 'radius_km': 6.0}, {'cbd': 'Cijantung Mal', 'lat': -6.324665, 'lon': 106.858936, 'radius_km': 2.0}, {'cbd': 'Cijantung Mall', 'lat': -6.321245, 'lon': 106.877178, 'radius_km': 2.9}, {'cbd': 'Cilandak Mall', 'lat': -6.333381, 'lon': 106.807429, 'radius_km': 5.4}, {'cbd': 'Cilandak Town Square', 'lat': -6.288456, 'lon': 106.801712, 'radius_km': 4.22}, {'cbd': 'CILEDUG PLAZA', 'lat': -6.219268, 'lon': 106.721608, 'radius_km': 1.0}, {'cbd': 'Cinere Bellevue Mall', 'lat': -6.316469, 'lon': 106.780519, 'radius_km': 0.5}, {'cbd': 'cinere mall', 'lat': -6.349428, 'lon': 106.805495, 'radius_km': 5.39}, {'cbd': 'Cipadu Trade Center', 'lat': -6.235184, 'lon': 106.751341, 'radius_km': 4.08}, {'cbd': 'Cipinang Indah Mall', 'lat': -6.25168, 'lon': 106.93546, 'radius_km': 5.3}, {'cbd': 'Cipinang Muara', 'lat': -6.228348, 'lon': 106.885422, 'radius_km': 3.0}, {'cbd': 'ciplaz cengkareng', 'lat': -6.155291, 'lon': 106.7491, 'radius_km': 3.4}, {'cbd': 'Ciplaz Ciledug', 'lat': -6.222296, 'lon': 106.743375, 'radius_km': 2.0}, {'cbd': 'Ciplaz Klender', 'lat': -6.209139, 'lon': 106.904601, 'radius_km': 2.5}, {'cbd': 'Ciplaz Klender (RAMAYANA)', 'lat': -6.22705, 'lon': 106.897397, 'radius_km': 2.48}, {'cbd': 'Ciputra Mall', 'lat': -6.165195, 'lon': 106.783875, 'radius_km': 0.5}, {'cbd': 'Ciputra Superblok', 'lat': -6.163879, 'lon': 106.727679, 'radius_km': 1.1}, {'cbd': 'CIPUTRA WORLD', 'lat': -6.226395, 'lon': 106.821394, 'radius_km': 0.55}, {'cbd': 'CityPlaza Jatinegara', 'lat': -6.210424, 'lon': 106.858475, 'radius_km': 0.79}, {'cbd': 'Citywalk Sudirman', 'lat': -6.208188, 'lon': 106.820875, 'radius_km': 0.93}, {'cbd': 'Citywalk Sudirman Jakarta', 'lat': -6.206681, 'lon': 106.81674, 'radius_km': 1.0}, {'cbd': 'Daerah Kalibata City', 'lat': -6.244715, 'lon': 106.868662, 'radius_km': 3.0}, {'cbd': 'Depok town square', 'lat': -6.351017, 'lon': 106.812611, 'radius_km': 6.0}, {'cbd': 'DM Mall', 'lat': -6.14172, 'lon': 106.698366, 'radius_km': 2.0}, {'cbd': 'emporium mall', 'lat': -6.133069, 'lon': 106.761766, 'radius_km': 3.0}, {'cbd': 'Emporium Pluit Mall', 'lat': -6.12554, 'lon': 106.794196, 'radius_km': 3.0}, {'cbd': 'Epicentrum XXI', 'lat': -6.208114, 'lon': 106.831816, 'radius_km': 2.0}, {'cbd': 'farmers family', 'lat': -6.2296, 'lon': 106.94452, 'radius_km': 1.0}, {'cbd': 'Farmers Family Pondok kopi', 'lat': -6.225994, 'lon': 106.942682, 'radius_km': 0.5}, {'cbd': 'FK UI', 'lat': -6.196139, 'lon': 106.849417, 'radius_km': 1.0}, {'cbd': 'Gajah Mada Plaza', 'lat': -6.158164, 'lon': 106.817901, 'radius_km': 1.2}, {'cbd': 'GANDARIA CITY', 'lat': -6.25115, 'lon': 106.773544, 'radius_km': 3.6}, {'cbd': 'Gandaria City Mall', 'lat': -6.249924, 'lon': 106.778016, 'radius_km': 2.36}, {'cbd': 'Gelanggang remaja', 'lat': -6.339621, 'lon': 106.881062, 'radius_km': 3.0}, {'cbd': 'glodok china town market', 'lat': -6.144608, 'lon': 106.810502, 'radius_km': 1.0}, {'cbd': 'Glodok Chinatown Market', 'lat': -6.145326, 'lon': 106.80773, 'radius_km': 1.0}, {'cbd': 'Glodok Plaza', 'lat': -6.144628, 'lon': 106.81059, 'radius_km': 1.0}, {'cbd': 'Golden Trully Mall', 'lat': -6.153219, 'lon': 106.836122, 'radius_km': 1.0}, {'cbd': 'Golden Truly Mall', 'lat': -6.159027, 'lon': 106.838485, 'radius_km': 3.0}, {'cbd': 'grade mall koja', 'lat': -6.110053, 'lon': 106.88557, 'radius_km': 1.0}, {'cbd': 'Gramedia Matraman', 'lat': -6.201717, 'lon': 106.851394, 'radius_km': 1.0}, {'cbd': 'grand cakung', 'lat': -6.193017, 'lon': 106.953123, 'radius_km': 6.0}, {'cbd': 'Grand Indonesia', 'lat': -6.202103, 'lon': 106.808833, 'radius_km': 1.66}, {'cbd': 'Grand ITC Permata Hijau', 'lat': -6.214348, 'lon': 106.780243, 'radius_km': 3.1}, {'cbd': 'Grand Paragon Mall', 'lat': -6.15253, 'lon': 106.81967, 'radius_km': 1.0}, {'cbd': 'grand sedayu', 'lat': -6.131907, 'lon': 106.728697, 'radius_km': 1.0}, {'cbd': 'Grandlucky - Green lake city', 'lat': -6.183728, 'lon': 106.700572, 'radius_km': 1.5}, {'cbd': 'Green Garden', 'lat': -6.17551, 'lon': 106.76561, 'radius_km': 1.0}, {'cbd': 'Green Pramuka', 'lat': -6.192692, 'lon': 106.865728, 'radius_km': 3.0}, {'cbd': 'Green pramuka mall', 'lat': -6.194808, 'lon': 106.883896, 'radius_km': 2.0}, {'cbd': 'Green Pramuka Square', 'lat': -6.184958, 'lon': 106.867106, 'radius_km': 1.9}, {'cbd': 'Green Pramuka Square Mall', 'lat': -6.192152, 'lon': 106.864407, 'radius_km': 2.82}, {'cbd': 'Green Sedayu Biz Park Cakung', 'lat': -6.131331, 'lon': 106.955587, 'radius_km': 3.4}, {'cbd': 'green sedayu mall', 'lat': -6.128704, 'lon': 106.726527, 'radius_km': 2.0}, {'cbd': 'Green Sendayu Mall', 'lat': -6.131543, 'lon': 106.716293, 'radius_km': 4.51}, {'cbd': 'Green Terace TMII', 'lat': -6.307177, 'lon': 106.888486, 'radius_km': 0.5}, {'cbd': 'Halim Perdana Kusuma', 'lat': -6.192886, 'lon': 106.875369, 'radius_km': 1.0}, {'cbd': 'Harco Mangga 2', 'lat': -6.139077, 'lon': 106.826936, 'radius_km': 0.5}, {'cbd': 'Harco Mangga Dua', 'lat': -6.147519, 'lon': 106.83907, 'radius_km': 2.0}, {'cbd': 'Harco Pasar Besar', 'lat': -6.165951, 'lon': 106.8316, 'radius_km': 1.0}, {'cbd': 'hari-hari duta harapan indah', 'lat': -6.147278, 'lon': 106.771555, 'radius_km': 2.0}, {'cbd': 'Hero Alfa Indah', 'lat': -6.218559, 'lon': 106.725366, 'radius_km': 3.0}, {'cbd': 'Hero Supermarket Alfa Indah', 'lat': -6.218279, 'lon': 106.752211, 'radius_km': 0.5}, {'cbd': 'hipermart puri', 'lat': -6.184099, 'lon': 106.740145, 'radius_km': 1.0}, {'cbd': 'ITC Cempaka Mas', 'lat': -6.169556, 'lon': 106.867885, 'radius_km': 3.0}, {'cbd': 'ITC Cipulir', 'lat': -6.240922, 'lon': 106.756499, 'radius_km': 4.17}, {'cbd': 'ITC Cipulir Mas', 'lat': -6.242366, 'lon': 106.756615, 'radius_km': 3.34}, {'cbd': 'ITC Fatmawati', 'lat': -6.262645, 'lon': 106.79425, 'radius_km': 2.0}, {'cbd': 'ITC Mangga Dua', 'lat': -6.119622, 'lon': 106.822245, 'radius_km': 2.7}, {'cbd': 'ITC Permata Hijau', 'lat': -6.215713, 'lon': 106.775154, 'radius_km': 2.43}, {'cbd': 'ITC Roxy Mas', 'lat': -6.165868, 'lon': 106.8083, 'radius_km': 2.62}, {'cbd': 'Jakarta International Stadium', 'lat': -6.131411, 'lon': 106.861389, 'radius_km': 1.0}, {'cbd': 'Jalan Jenderal Sudirman', 'lat': -6.206349, 'lon': 106.815087, 'radius_km': 1.0}, {'cbd': 'Jalan Metro Pondok Indah', 'lat': -6.254919, 'lon': 106.78232, 'radius_km': 1.0}, {'cbd': 'Jalan Perjuangan', 'lat': -6.194606, 'lon': 106.769081, 'radius_km': 1.0}, {'cbd': 'Jatijajar', 'lat': -6.327378, 'lon': 106.86412, 'radius_km': 5.0}, {'cbd': 'JIEXPO', 'lat': -6.169133, 'lon': 106.848, 'radius_km': 6.0}, {'cbd': 'kalibata city', 'lat': -6.275191, 'lon': 106.82951, 'radius_km': 6.0}, {'cbd': 'Kalibata City Square', 'lat': -6.250291, 'lon': 106.840769, 'radius_km': 1.0}, {'cbd': 'Kalibata Mall', 'lat': -6.247303, 'lon': 106.841262, 'radius_km': 3.0}, {'cbd': 'Kantor Walikota Jakarta Barat', 'lat': -6.185019, 'lon': 106.733633, 'radius_km': 0.5}, {'cbd': 'Kawasan AEON MALL Tanjung Barat', 'lat': -6.303823, 'lon': 106.842488, 'radius_km': 1.0}, {'cbd': 'Kawasan AEON Tanjung Barat', 'lat': -6.307093, 'lon': 106.845476, 'radius_km': 1.0}, {'cbd': 'Kawasan AEON Tj Barat', 'lat': -6.307065, 'lon': 106.832935, 'radius_km': 3.3}, {'cbd': 'Kawasan Atrium', 'lat': -6.165469, 'lon': 106.848047, 'radius_km': 4.6}, {'cbd': 'Kawasan Beltway Office Park', 'lat': -6.278717, 'lon': 106.813002, 'radius_km': 2.0}, {'cbd': 'Kawasan Bintaro', 'lat': -6.272673, 'lon': 106.760062, 'radius_km': 2.87}, {'cbd': 'Kawasan Bintaro Jaya', 'lat': -6.248304, 'lon': 106.763183, 'radius_km': 4.37}, {'cbd': 'Kawasan Bintaro Sektor 3', 'lat': -6.26965, 'lon': 106.756482, 'radius_km': 2.44}, {'cbd': 'Kawasan Bintaro Sektor 3A', 'lat': -6.27314, 'lon': 106.761265, 'radius_km': 2.77}, {'cbd': 'Kawasan Blok M', 'lat': -6.248797, 'lon': 106.809222, 'radius_km': 3.36}, {'cbd': 'Kawasan Blok M Square', 'lat': -6.239353, 'lon': 106.811072, 'radius_km': 2.0}, {'cbd': 'Kawasan Centraland Cengkareng', 'lat': -6.154898, 'lon': 106.744377, 'radius_km': 3.0}, {'cbd': 'Kawasan Centraland Jakarta', 'lat': -6.155027, 'lon': 106.746193, 'radius_km': 3.0}, {'cbd': 'Kawasan Cibis Park', 'lat': -6.302718, 'lon': 106.82091, 'radius_km': 3.84}, {'cbd': 'Kawasan Cibis Tower', 'lat': -6.302589, 'lon': 106.826876, 'radius_km': 5.0}, {'cbd': 'Kawasan Cibis Tower CBD', 'lat': -6.3012, 'lon': 106.828778, 'radius_km': 1.99}, {'cbd': 'Kawasan Cikini', 'lat': -6.183892, 'lon': 106.843831, 'radius_km': 1.0}, {'cbd': 'Kawasan Cilandak Town Square', 'lat': -6.30075, 'lon': 106.802057, 'radius_km': 3.73}, {'cbd': 'Kawasan Cilandak Town Square (CITOS)', 'lat': -6.289888, 'lon': 106.801747, 'radius_km': 1.0}, {'cbd': 'Kawasan Ciplaz Ciledug', 'lat': -6.224683, 'lon': 106.743643, 'radius_km': 3.64}, {'cbd': 'Kawasan Citos', 'lat': -6.289025, 'lon': 106.812009, 'radius_km': 3.89}, {'cbd': 'Kawasan Depok Town Square', 'lat': -6.351017, 'lon': 106.81261, 'radius_km': 6.0}, {'cbd': 'Kawasan Dharmawangsa Square', 'lat': -6.254692, 'lon': 106.804545, 'radius_km': 1.0}, {'cbd': 'Kawasan Fatmawati', 'lat': -6.302071, 'lon': 106.797098, 'radius_km': 3.29}, {'cbd': 'Kawasan Gandaria City', 'lat': -6.245235, 'lon': 106.760165, 'radius_km': 6.0}, {'cbd': 'Kawasan Giant Ciledug', 'lat': -6.232607, 'lon': 106.753025, 'radius_km': 1.5}, {'cbd': 'Kawasan Gran ITC Permata Hijau', 'lat': -6.220462, 'lon': 106.785075, 'radius_km': 0.77}, {'cbd': 'Kawasan ITC Cipulir', 'lat': -6.240635, 'lon': 106.767745, 'radius_km': 1.67}, {'cbd': 'Kawasan ITC Fatmawati', 'lat': -6.274286, 'lon': 106.81085, 'radius_km': 3.7}, {'cbd': 'Kawasan ITC Mangga Dua', 'lat': -6.162066, 'lon': 106.843733, 'radius_km': 4.0}, {'cbd': 'Kawasan Jakarta Convention Center', 'lat': -6.208771, 'lon': 106.811779, 'radius_km': 2.0}, {'cbd': 'Kawasan Jakarta Design Center', 'lat': -6.198774, 'lon': 106.804992, 'radius_km': 1.9}, {'cbd': 'Kawasan Kalibata', 'lat': -6.25446, 'lon': 106.838741, 'radius_km': 1.56}, {'cbd': 'Kawasan Kalibata City', 'lat': -6.260906, 'lon': 106.834862, 'radius_km': 3.46}, {'cbd': 'Kawasan Klender', 'lat': -6.304714, 'lon': 106.872292, 'radius_km': 2.0}, {'cbd': 'Kawasan Kota Casablanca', 'lat': -6.241403, 'lon': 106.844874, 'radius_km': 2.67}, {'cbd': 'Kawasan Kota Kasablanka', 'lat': -6.238231, 'lon': 106.841909, 'radius_km': 5.68}, {'cbd': 'Kawasan Kuningan City', 'lat': -6.234312, 'lon': 106.855625, 'radius_km': 3.8}, {'cbd': 'Kawasan Lebak Bulus Grab', 'lat': -6.299983, 'lon': 106.776649, 'radius_km': 2.9}, {'cbd': 'Kawasan Lenteng Agung', 'lat': -6.347619, 'lon': 106.81251, 'radius_km': 4.6}, {'cbd': 'Kawasan Lippa Mall Kemang', 'lat': -6.266396, 'lon': 106.815858, 'radius_km': 1.85}, {'cbd': 'Kawasan Lippo Mall Kemang', 'lat': -6.26504, 'lon': 106.812141, 'radius_km': 3.26}, {'cbd': 'Kawasan LOTTE GROS Pasar Rebo', 'lat': -6.30272, 'lon': 106.853456, 'radius_km': 2.0}, {'cbd': 'Kawasan lotte mart fatmawati', 'lat': -6.280217, 'lon': 106.802635, 'radius_km': 1.0}, {'cbd': 'Kawasan Margocity', 'lat': -6.353262, 'lon': 106.815015, 'radius_km': 6.0}, {'cbd': 'Kawasan Mega Kuningan', 'lat': -6.238324, 'lon': 106.822885, 'radius_km': 2.18}, {'cbd': 'Kawasan Pasar Besar Lenteng Agung', 'lat': -6.338845, 'lon': 106.82236, 'radius_km': 3.0}, {'cbd': 'Kawasan Pasar Cijantung', 'lat': -6.306246, 'lon': 106.854921, 'radius_km': 1.02}, {'cbd': 'Kawasan Pasar Grosir Cililitan', 'lat': -6.274333, 'lon': 106.877336, 'radius_km': 3.0}, {'cbd': 'Kawasan Pasar Lenteng Agung', 'lat': -6.340341, 'lon': 106.813336, 'radius_km': 4.14}, {'cbd': 'Kawasan pasar rebo cijantung', 'lat': -6.297496, 'lon': 106.848001, 'radius_km': 2.85}, {'cbd': 'kawasan Pejanten Village Mall Plaza', 'lat': -6.278734, 'lon': 106.825857, 'radius_km': 0.67}, {'cbd': 'Kawasan Pejaten Raya', 'lat': -6.285179, 'lon': 106.826179, 'radius_km': 1.0}, {'cbd': 'Kawasan Pejaten Village', 'lat': -6.276617, 'lon': 106.82466, 'radius_km': 1.0}, {'cbd': 'Kawasan Pelabuhan Tanjung Priok', 'lat': -6.1056, 'lon': 106.93039, 'radius_km': 1.0}, {'cbd': 'Kawasan Pondok Indah', 'lat': -6.275578, 'lon': 106.788773, 'radius_km': 3.94}, {'cbd': 'Kawasan Pondok Indah Mall', 'lat': -6.257876, 'lon': 106.760929, 'radius_km': 5.7}, {'cbd': 'Kawasan Pondok Kelapa Town Square', 'lat': -6.244345, 'lon': 106.939851, 'radius_km': 2.01}, {'cbd': 'Kawasan pondon indah jakarta', 'lat': -6.284155, 'lon': 106.789099, 'radius_km': 3.0}, {'cbd': 'Kawasan Radio Dalam Square', 'lat': -6.261137, 'lon': 106.787759, 'radius_km': 1.0}, {'cbd': 'Kawasan Ramayanan Kebayoran Lama', 'lat': -6.236094, 'lon': 106.758309, 'radius_km': 5.0}, {'cbd': 'Kawasan RS Dr Suyoto', 'lat': -6.279872, 'lon': 106.76252, 'radius_km': 1.9}, {'cbd': 'Kawasan RS Siloam Asri', 'lat': -6.267996, 'lon': 106.831089, 'radius_km': 2.0}, {'cbd': 'Kawasan Senayan', 'lat': -6.226015, 'lon': 106.78625, 'radius_km': 3.47}, {'cbd': 'Kawasan Senayan City', 'lat': -6.229409, 'lon': 106.808494, 'radius_km': 2.34}, {'cbd': 'Kawasan Senayan GBK', 'lat': -6.229011, 'lon': 106.790314, 'radius_km': 1.0}, {'cbd': 'Kawasan Simpang Jl TB Simatupang', 'lat': -6.287761, 'lon': 106.803631, 'radius_km': 2.0}, {'cbd': 'Kawasan Simpang Lenteng Agung', 'lat': -6.309627, 'lon': 106.828376, 'radius_km': 2.0}, {'cbd': 'Kawasan Sudirman Hill Residence', 'lat': -6.20925, 'lon': 106.807236, 'radius_km': 1.0}, {'cbd': 'Kawasan Tamansari Hive', 'lat': -6.236812, 'lon': 106.875126, 'radius_km': 0.92}, {'cbd': 'Kawasan TB Simatupang', 'lat': -6.291796, 'lon': 106.811375, 'radius_km': 1.0}, {'cbd': 'Kawasan The Dharmawangsa Square', 'lat': -6.257319, 'lon': 106.804825, 'radius_km': 1.0}, {'cbd': 'Kawasan The Hive DI Panjaitan', 'lat': -6.240778, 'lon': 106.874806, 'radius_km': 0.5}, {'cbd': 'Kawasan Transmart Cilandak', 'lat': -6.31305, 'lon': 106.816071, 'radius_km': 2.0}, {'cbd': 'Kawasan Universitas 17 Agustus', 'lat': -6.140196, 'lon': 106.861414, 'radius_km': 2.0}, {'cbd': 'Kawasan Universitas Negeri Jakarta', 'lat': -6.192162, 'lon': 106.866951, 'radius_km': 2.0}, {'cbd': 'KawasanCitos (Cilandak Town Square)', 'lat': -6.295542, 'lon': 106.793039, 'radius_km': 2.0}, {'cbd': 'Kawsan Pondok Indah', 'lat': -6.294781, 'lon': 106.79998, 'radius_km': 2.97}, {'cbd': 'KBN Cakung', 'lat': -6.14758, 'lon': 106.9526, 'radius_km': 2.0}, {'cbd': 'KBN Marunda', 'lat': -6.10994, 'lon': 106.95543, 'radius_km': 1.0}, {'cbd': 'Kedoya Green Garden', 'lat': -6.161675, 'lon': 106.763142, 'radius_km': 1.0}, {'cbd': 'KELAPA GADING TRADE CENTRE', 'lat': -6.159642, 'lon': 106.882236, 'radius_km': 1.0}, {'cbd': 'Kemang Square', 'lat': -6.269263, 'lon': 106.830736, 'radius_km': 3.6}, {'cbd': 'Kemang Village', 'lat': -6.252092, 'lon': 106.816662, 'radius_km': 2.0}, {'cbd': 'Kencana Tower', 'lat': -6.199317, 'lon': 106.761749, 'radius_km': 0.5}, {'cbd': 'Koja Trade Mall', 'lat': -6.115577, 'lon': 106.926684, 'radius_km': 2.8}, {'cbd': 'Kota Casablanca', 'lat': -6.230105, 'lon': 106.844151, 'radius_km': 0.8}, {'cbd': 'Kota Casablanka', 'lat': -6.213087, 'lon': 106.845427, 'radius_km': 1.1}, {'cbd': 'Kota Kasablanka', 'lat': -6.221248, 'lon': 106.849259, 'radius_km': 3.0}, {'cbd': 'Kota Tua', 'lat': -6.134582, 'lon': 106.809896, 'radius_km': 0.5}, {'cbd': 'Kuningan', 'lat': -6.202955, 'lon': 106.840577, 'radius_km': 2.7}, {'cbd': 'Kuningan City', 'lat': -6.243926, 'lon': 106.826932, 'radius_km': 2.9}, {'cbd': 'La Piazza', 'lat': -6.166412, 'lon': 106.91711, 'radius_km': 1.0}, {'cbd': 'Landmark Pluit', 'lat': -6.124625, 'lon': 106.797942, 'radius_km': 0.7}, {'cbd': 'Lippo Mal Puri', 'lat': -6.207456, 'lon': 106.746253, 'radius_km': 2.0}, {'cbd': 'Lippo Mall Kemang', 'lat': -6.258306, 'lon': 106.815165, 'radius_km': 4.1}, {'cbd': 'Lippo mall nusantara', 'lat': -6.21524, 'lon': 106.813645, 'radius_km': 1.5}, {'cbd': 'Lippo Mall Puri', 'lat': -6.183482, 'lon': 106.742975, 'radius_km': 3.0}, {'cbd': 'lippo mall puri indah', 'lat': -6.197809, 'lon': 106.738524, 'radius_km': 2.27}, {'cbd': 'Lippo Plaza Kramat Jati', 'lat': -6.279883, 'lon': 106.87091, 'radius_km': 2.14}, {'cbd': 'Lippo Puri Indah Mall', 'lat': -6.200238, 'lon': 106.731339, 'radius_km': 1.0}, {'cbd': 'Lippo Puri Mall', 'lat': -6.187382, 'lon': 106.759041, 'radius_km': 2.87}, {'cbd': 'Lokasari Square', 'lat': -6.148485, 'lon': 106.826415, 'radius_km': 0.57}, {'cbd': 'Lotte Grosir Kelapa Gading', 'lat': -6.156248, 'lon': 106.896635, 'radius_km': 1.03}, {'cbd': 'Lotte Grosir Meruya', 'lat': -6.198439, 'lon': 106.733312, 'radius_km': 0.97}, {'cbd': 'lotte grosir pasar rebo', 'lat': -6.308252, 'lon': 106.867685, 'radius_km': 0.5}, {'cbd': 'Lotte Mall Jakarta', 'lat': -6.221532, 'lon': 106.82369, 'radius_km': 0.95}, {'cbd': 'Lotte Mart Fatmawati', 'lat': -6.279212, 'lon': 106.796861, 'radius_km': 1.0}, {'cbd': 'Lotte Mart Taman Surya', 'lat': -6.127257, 'lon': 106.705359, 'radius_km': 2.64}, {'cbd': 'Lotte Meruya', 'lat': -6.211804, 'lon': 106.729976, 'radius_km': 2.0}, {'cbd': 'LTC Glodok', 'lat': -6.149929, 'lon': 106.812754, 'radius_km': 0.61}, {'cbd': 'LTC Glodok Hayam Wuruk', 'lat': -6.148223, 'lon': 106.811214, 'radius_km': 1.0}, {'cbd': 'Lulu Hypermart', 'lat': -6.195163, 'lon': 106.950218, 'radius_km': 2.2}, {'cbd': 'M Blok Space', 'lat': -6.251746, 'lon': 106.79233, 'radius_km': 2.0}, {'cbd': 'Mal Atrium Senen', 'lat': -6.176978, 'lon': 106.840445, 'radius_km': 0.5}, {'cbd': 'Mal Cijantung', 'lat': -6.326734, 'lon': 106.864431, 'radius_km': 2.8}, {'cbd': 'Mal Ciputra Jakarta', 'lat': -6.16548, 'lon': 106.791383, 'radius_km': 1.0}, {'cbd': 'Mal Kota Kasablanka', 'lat': -6.241789, 'lon': 106.84633, 'radius_km': 2.0}, {'cbd': 'Mall AEON tanjung barat', 'lat': -6.310416, 'lon': 106.827063, 'radius_km': 5.38}, {'cbd': 'Mall Ambassador', 'lat': -6.223184, 'lon': 106.82704, 'radius_km': 0.5}, {'cbd': 'Mall Arta Gading', 'lat': -6.12916, 'lon': 106.893413, 'radius_km': 2.0}, {'cbd': 'Mall Artha Gading', 'lat': -6.148491, 'lon': 106.885668, 'radius_km': 3.26}, {'cbd': 'mall atha gading', 'lat': -6.122351, 'lon': 106.875968, 'radius_km': 2.0}, {'cbd': 'mall atrium', 'lat': -6.1775, 'lon': 106.84016, 'radius_km': 1.0}, {'cbd': 'Mall Atrium Senen', 'lat': -6.176427, 'lon': 106.841206, 'radius_km': 1.0}, {'cbd': 'Mall Bassura', 'lat': -6.228434, 'lon': 106.874402, 'radius_km': 4.0}, {'cbd': 'Mall Basura', 'lat': -6.227639, 'lon': 106.874417, 'radius_km': 2.54}, {'cbd': 'Mall Bella Tera', 'lat': -6.17371, 'lon': 106.89994, 'radius_km': 1.0}, {'cbd': 'Mall Bellagio Boutique', 'lat': -6.229913, 'lon': 106.824555, 'radius_km': 1.0}, {'cbd': 'mall central park', 'lat': -6.17885, 'lon': 106.77356, 'radius_km': 2.0}, {'cbd': 'Mall Cijantung', 'lat': -6.313224, 'lon': 106.873327, 'radius_km': 4.38}, {'cbd': 'MALL CINERE', 'lat': -6.348248, 'lon': 106.795114, 'radius_km': 3.8}, {'cbd': 'Mall Cipinang Indah', 'lat': -6.239047, 'lon': 106.908, 'radius_km': 5.0}, {'cbd': 'Mall Ciputra', 'lat': -6.16218, 'lon': 106.775883, 'radius_km': 3.0}, {'cbd': 'mall ciputra (citraland)', 'lat': -6.155659, 'lon': 106.752878, 'radius_km': 2.0}, {'cbd': 'Mall Ciputra Jakarta', 'lat': -6.164528, 'lon': 106.784057, 'radius_km': 0.5}, {'cbd': 'Mall Daan Mogot', 'lat': -6.155731, 'lon': 106.712015, 'radius_km': 3.0}, {'cbd': 'mall daan mogot cengkareng', 'lat': -6.158052, 'lon': 106.708884, 'radius_km': 3.0}, {'cbd': 'Mall Gajah Mada', 'lat': -6.169509, 'lon': 106.814343, 'radius_km': 5.0}, {'cbd': 'Mall Gandaria City', 'lat': -6.248009, 'lon': 106.778605, 'radius_km': 6.0}, {'cbd': 'Mall Graha Cijantung', 'lat': -6.314517, 'lon': 106.875444, 'radius_km': 2.7}, {'cbd': 'Mall Grand Cakung', 'lat': -6.184866, 'lon': 106.951161, 'radius_km': 2.4}, {'cbd': 'mall green terrace TMII', 'lat': -6.310046, 'lon': 106.887541, 'radius_km': 1.0}, {'cbd': 'Mall ITC Roxy Mas', 'lat': -6.1667, 'lon': 106.81227, 'radius_km': 1.0}, {'cbd': 'Mall Kelapa Gading', 'lat': -6.1594, 'lon': 106.91424, 'radius_km': 2.44}, {'cbd': 'Mall Kelapa Hading', 'lat': -6.15226, 'lon': 106.91543, 'radius_km': 1.0}, {'cbd': 'Mall Koja', 'lat': -6.114254, 'lon': 106.899016, 'radius_km': 2.6}, {'cbd': 'Mall Matahari Daan Mogot', 'lat': -6.134308, 'lon': 106.70003, 'radius_km': 2.68}, {'cbd': 'mall metro cipulir', 'lat': -6.22542, 'lon': 106.74643, 'radius_km': 3.0}, {'cbd': 'Mall Neo Soho', 'lat': -6.170339, 'lon': 106.787644, 'radius_km': 1.0}, {'cbd': 'Mall Of Indonesia', 'lat': -6.15074, 'lon': 106.89477, 'radius_km': 1.02}, {'cbd': 'Mall PIK Avenue', 'lat': -6.10473, 'lon': 106.74893, 'radius_km': 1.0}, {'cbd': 'mall plaza semanggj', 'lat': -6.238735, 'lon': 106.829068, 'radius_km': 4.0}, {'cbd': 'mall pondok indah', 'lat': -6.276064, 'lon': 106.781927, 'radius_km': 2.24}, {'cbd': 'Mall Pondok Kelapa Town Square', 'lat': -6.229374, 'lon': 106.935242, 'radius_km': 1.2}, {'cbd': 'MALL PURI', 'lat': -6.188052, 'lon': 106.73285, 'radius_km': 1.0}, {'cbd': 'mall puri indah', 'lat': -6.187185, 'lon': 106.7348, 'radius_km': 4.0}, {'cbd': 'Mall Slipi Jaya', 'lat': -6.189181, 'lon': 106.787289, 'radius_km': 1.0}, {'cbd': 'Mall Sunter', 'lat': -6.137531, 'lon': 106.867654, 'radius_km': 2.0}, {'cbd': 'mall taman anggrek', 'lat': -6.187103, 'lon': 106.782936, 'radius_km': 2.97}, {'cbd': 'MALL TAMAN ANGREK', 'lat': -6.185005, 'lon': 106.782869, 'radius_km': 1.0}, {'cbd': 'Mall Taman Palem', 'lat': -6.135776, 'lon': 106.746323, 'radius_km': 3.1}, {'cbd': 'Mampang Business Park', 'lat': -6.255905, 'lon': 106.828336, 'radius_km': 1.3}, {'cbd': 'Mangga 2 Square', 'lat': -6.147427, 'lon': 106.838613, 'radius_km': 2.26}, {'cbd': 'MANGGA DUA', 'lat': -6.147558, 'lon': 106.839105, 'radius_km': 1.7}, {'cbd': 'mangga dua mall', 'lat': -6.129718, 'lon': 106.82488, 'radius_km': 1.84}, {'cbd': 'Mangga dua square', 'lat': -6.145244, 'lon': 106.836233, 'radius_km': 4.24}, {'cbd': 'Manhattan', 'lat': -6.068837, 'lon': 106.701717, 'radius_km': 1.0}, {'cbd': 'Margo City', 'lat': -6.35452, 'lon': 106.807364, 'radius_km': 3.65}, {'cbd': 'margo city mall', 'lat': -6.338918, 'lon': 106.836627, 'radius_km': 4.0}, {'cbd': 'Marunda Center', 'lat': -6.113037, 'lon': 106.969979, 'radius_km': 2.4}, {'cbd': 'Marunda Centre', 'lat': -6.10578, 'lon': 106.96264, 'radius_km': 2.0}, {'cbd': 'Matahari Mall Daan Mogot Baru', 'lat': -6.142997, 'lon': 106.708245, 'radius_km': 2.9}, {'cbd': 'Mega Kuningan', 'lat': -6.230202, 'lon': 106.824224, 'radius_km': 2.98}, {'cbd': 'Mega Kuningan City', 'lat': -6.247716, 'lon': 106.829785, 'radius_km': 3.25}, {'cbd': 'Menteng Huis', 'lat': -6.189904, 'lon': 106.838798, 'radius_km': 4.0}, {'cbd': 'Metro Atom Plaza', 'lat': -6.160338, 'lon': 106.82338, 'radius_km': 1.5}, {'cbd': 'Metro Kebayoran', 'lat': -6.241663, 'lon': 106.764935, 'radius_km': 1.0}, {'cbd': 'metro sunter plaza', 'lat': -6.133174, 'lon': 106.870432, 'radius_km': 1.0}, {'cbd': 'Metropole XXI Cikini', 'lat': -6.204044, 'lon': 106.841397, 'radius_km': 1.49}, {'cbd': 'MGK Mall Kemayoran', 'lat': -6.157843, 'lon': 106.841843, 'radius_km': 1.16}, {'cbd': 'MOI', 'lat': -6.149911, 'lon': 106.873746, 'radius_km': 6.0}, {'cbd': 'MOI KELAPA GADING', 'lat': -6.151441, 'lon': 106.883343, 'radius_km': 3.8}, {'cbd': 'Naga Swalayan', 'lat': -6.207539, 'lon': 106.97778, 'radius_km': 1.6}, {'cbd': 'Office Park TB Simatupang', 'lat': -6.325219, 'lon': 106.82588, 'radius_km': 4.0}, {'cbd': 'one bell park', 'lat': -6.307854, 'lon': 106.793934, 'radius_km': 0.5}, {'cbd': 'one bellpark mall', 'lat': -6.307376, 'lon': 106.799157, 'radius_km': 1.82}, {'cbd': 'One Belpark', 'lat': -6.306884, 'lon': 106.796036, 'radius_km': 0.5}, {'cbd': 'One Belpark Mall', 'lat': -6.308583, 'lon': 106.795635, 'radius_km': 0.98}, {'cbd': 'Orion Sports Center', 'lat': -6.136253, 'lon': 106.805818, 'radius_km': 1.0}, {'cbd': 'pacific place', 'lat': -6.23907, 'lon': 106.82893, 'radius_km': 3.0}, {'cbd': 'Pacific Place Mall', 'lat': -6.239311, 'lon': 106.824244, 'radius_km': 2.99}, {'cbd': 'Pantai Ancol', 'lat': -6.12205, 'lon': 106.85601, 'radius_km': 1.0}, {'cbd': 'Pantai Qncol', 'lat': -6.12289, 'lon': 106.8579, 'radius_km': 1.0}, {'cbd': 'PASAR ANYAR BAHARI', 'lat': -6.121245, 'lon': 106.876561, 'radius_km': 1.0}, {'cbd': 'Pasar Baru', 'lat': -6.164923, 'lon': 106.832702, 'radius_km': 1.0}, {'cbd': 'Pasar Benhil', 'lat': -6.206733, 'lon': 106.805958, 'radius_km': 2.0}, {'cbd': 'Pasar Besar Lenteng Agung', 'lat': -6.325782, 'lon': 106.823958, 'radius_km': 1.46}, {'cbd': 'Pasar Cempaka Putih', 'lat': -6.177083, 'lon': 106.8685, 'radius_km': 3.0}, {'cbd': 'Pasar Cengkareeng', 'lat': -6.147511, 'lon': 106.728711, 'radius_km': 0.5}, {'cbd': 'Pasar Cengkareng', 'lat': -6.164415, 'lon': 106.728223, 'radius_km': 1.4}, {'cbd': 'Pasar Cibubur', 'lat': -6.349578, 'lon': 106.867551, 'radius_km': 1.3}, {'cbd': 'Pasar Cikini', 'lat': -6.202301, 'lon': 106.841986, 'radius_km': 1.0}, {'cbd': 'Pasar Cipinang Muara', 'lat': -6.226653, 'lon': 106.885887, 'radius_km': 1.8}, {'cbd': 'Pasar Dadap', 'lat': -6.09581, 'lon': 106.7184, 'radius_km': 1.0}, {'cbd': 'Pasar Gaplok', 'lat': -6.182225, 'lon': 106.843866, 'radius_km': 0.6}, {'cbd': 'Pasar Gembrong Baru', 'lat': -6.179333, 'lon': 106.858806, 'radius_km': 0.5}, {'cbd': 'pasar gembrong lama', 'lat': -6.17724, 'lon': 106.855365, 'radius_km': 1.0}, {'cbd': 'PASAR IKAN LUAR BATANG', 'lat': -6.127925, 'lon': 106.819109, 'radius_km': 1.82}, {'cbd': 'Pasar induk Kramat Jati', 'lat': -6.287067, 'lon': 106.864735, 'radius_km': 3.72}, {'cbd': 'Pasar Induk Kramatjati', 'lat': -6.29034, 'lon': 106.860208, 'radius_km': 2.5}, {'cbd': 'Pasar Jasa Mulya', 'lat': -6.209342, 'lon': 106.940152, 'radius_km': 0.5}, {'cbd': 'Pasar Jaya Cawang Kavling', 'lat': -6.243679, 'lon': 106.872651, 'radius_km': 1.0}, {'cbd': 'Pasar Jaya Cengkareng', 'lat': -6.14767, 'lon': 106.72891, 'radius_km': 0.5}, {'cbd': 'Pasar Jaya Cibubur', 'lat': -6.349018, 'lon': 106.874265, 'radius_km': 1.75}, {'cbd': 'Pasar Jaya Ciracas', 'lat': -6.337097, 'lon': 106.857167, 'radius_km': 1.88}, {'cbd': 'Pasar Jaya Enjo', 'lat': -6.205465, 'lon': 106.873482, 'radius_km': 1.03}, {'cbd': 'Pasar Jaya Pesanggrahan', 'lat': -6.256391, 'lon': 106.756397, 'radius_km': 0.55}, {'cbd': 'Pasar Kamal', 'lat': -6.09823, 'lon': 106.714424, 'radius_km': 1.0}, {'cbd': 'Pasar Kedoya', 'lat': -6.166817, 'lon': 106.773285, 'radius_km': 1.4}, {'cbd': 'Pasar Kemayoran', 'lat': -6.169218, 'lon': 106.854898, 'radius_km': 2.8}, {'cbd': 'Pasar Kenari Lama', 'lat': -6.190393, 'lon': 106.843752, 'radius_km': 0.75}, {'cbd': 'Pasar Kenari Mas', 'lat': -6.18553, 'lon': 106.858211, 'radius_km': 1.7}, {'cbd': 'Pasar Koja Baru', 'lat': -6.118065, 'lon': 106.915211, 'radius_km': 1.0}, {'cbd': 'Pasar Kombongan', 'lat': -6.156581, 'lon': 106.841261, 'radius_km': 1.18}, {'cbd': 'Pasar Kopro', 'lat': -6.175631, 'lon': 106.782122, 'radius_km': 1.0}, {'cbd': 'Pasar Kwitang Dalam', 'lat': -6.183374, 'lon': 106.840656, 'radius_km': 0.5}, {'cbd': 'Pasar Lenteng Agung', 'lat': -6.32146, 'lon': 106.826735, 'radius_km': 1.83}, {'cbd': 'PASAR LONTAR', 'lat': -6.113074, 'lon': 106.905897, 'radius_km': 1.0}, {'cbd': 'Pasar Maja Rawabadak', 'lat': -6.120403, 'lon': 106.904665, 'radius_km': 1.0}, {'cbd': 'Pasar Malaka', 'lat': -6.13198, 'lon': 106.95528, 'radius_km': 1.0}, {'cbd': 'pasar Mampang Prapatan', 'lat': -6.2477, 'lon': 106.826943, 'radius_km': 1.0}, {'cbd': 'pasar mandiri kelapa gading', 'lat': -6.162527, 'lon': 106.908332, 'radius_km': 1.0}, {'cbd': 'Pasar Mede', 'lat': -6.279246, 'lon': 106.79702, 'radius_km': 0.8}, {'cbd': 'Pasar Meruya Ilir', 'lat': -6.214311, 'lon': 106.739434, 'radius_km': 3.5}, {'cbd': 'Pasar Muara Angke', 'lat': -6.11125, 'lon': 106.762466, 'radius_km': 1.17}, {'cbd': 'Pasar Munjul', 'lat': -6.35619, 'lon': 106.894271, 'radius_km': 0.95}, {'cbd': 'Pasar Nangka Bungur', 'lat': -6.170563, 'lon': 106.851629, 'radius_km': 1.0}, {'cbd': 'PASAR OKAN LUAR BATANG', 'lat': -6.127249, 'lon': 106.813436, 'radius_km': 1.0}, {'cbd': 'Pasar Pagi Kayu Putih', 'lat': -6.181265, 'lon': 106.893508, 'radius_km': 1.8}, {'cbd': 'Pasar Palmerah', 'lat': -6.207368, 'lon': 106.790753, 'radius_km': 0.93}, {'cbd': 'Pasar Paseban', 'lat': -6.192715, 'lon': 106.8587, 'radius_km': 2.0}, {'cbd': 'Pasar Patra', 'lat': -6.179729, 'lon': 106.772446, 'radius_km': 1.3}, {'cbd': 'PASAR PELITA', 'lat': -6.127979, 'lon': 106.891733, 'radius_km': 1.0}, {'cbd': 'Pasar Perumnas Klender', 'lat': -6.221568, 'lon': 106.929001, 'radius_km': 1.0}, {'cbd': 'Pasar Poncol', 'lat': -6.174817, 'lon': 106.846941, 'radius_km': 0.5}, {'cbd': 'Pasar Pondok Bambu', 'lat': -6.234665, 'lon': 106.913135, 'radius_km': 2.29}, {'cbd': 'Pasar Pondok Gede', 'lat': -6.299373, 'lon': 106.909467, 'radius_km': 3.12}, {'cbd': 'Pasar Pramuka', 'lat': -6.197462, 'lon': 106.860058, 'radius_km': 0.96}, {'cbd': 'Pasar Pulo Jahe', 'lat': -6.198655, 'lon': 106.929976, 'radius_km': 0.83}, {'cbd': 'Pasar Pulogadung', 'lat': -6.190516, 'lon': 106.901216, 'radius_km': 1.8}, {'cbd': 'Pasar Puri Indah', 'lat': -6.186385, 'lon': 106.755859, 'radius_km': 1.49}, {'cbd': 'Pasar Rakyat Kosambi Baru', 'lat': -6.174219, 'lon': 106.699443, 'radius_km': 3.0}, {'cbd': 'PASAR RAWA BADAK', 'lat': -6.112276, 'lon': 106.891068, 'radius_km': 1.0}, {'cbd': 'pasar rawasari', 'lat': -6.18257, 'lon': 106.871812, 'radius_km': 1.0}, {'cbd': 'Pasar Raya Manggarai', 'lat': -6.20215, 'lon': 106.8528, 'radius_km': 2.0}, {'cbd': 'Pasar Rumput', 'lat': -6.203419, 'lon': 106.841752, 'radius_km': 1.18}, {'cbd': 'Pasar Senen', 'lat': -6.16814, 'lon': 106.846056, 'radius_km': 2.88}, {'cbd': 'Pasar Senen Jaya', 'lat': -6.17151, 'lon': 106.844209, 'radius_km': 2.8}, {'cbd': 'PASAR SERDANG', 'lat': -6.149508, 'lon': 106.873845, 'radius_km': 1.0}, {'cbd': 'Pasar Suka Pura', 'lat': -6.14127, 'lon': 106.921511, 'radius_km': 1.0}, {'cbd': 'Pasar Sukapura', 'lat': -6.141372, 'lon': 106.921779, 'radius_km': 2.0}, {'cbd': 'Pasar Sumur Batu', 'lat': -6.16, 'lon': 106.870409, 'radius_km': 1.1}, {'cbd': 'PASAR SUNTER PODOMORO', 'lat': -6.142747, 'lon': 106.862353, 'radius_km': 1.0}, {'cbd': 'Pasar Tanah Abang', 'lat': -6.177717, 'lon': 106.816727, 'radius_km': 2.6}, {'cbd': 'Pasar Tanah Abang Blok A', 'lat': -6.186942, 'lon': 106.818277, 'radius_km': 1.1}, {'cbd': 'Pasar Timbul', 'lat': -6.338987, 'lon': 106.806162, 'radius_km': 2.24}, {'cbd': 'Pasar Tubun', 'lat': -6.202445, 'lon': 106.896401, 'radius_km': 0.8}, {'cbd': 'PASAR TUGU KRAMAT JAYA', 'lat': -6.129876, 'lon': 106.910083, 'radius_km': 1.0}, {'cbd': 'Pasar Ular', 'lat': -6.12986, 'lon': 106.893358, 'radius_km': 0.92}, {'cbd': 'Pasar Uler', 'lat': -6.12743, 'lon': 106.88898, 'radius_km': 1.0}, {'cbd': 'Pasaraya Blok M', 'lat': -6.246434, 'lon': 106.804877, 'radius_km': 1.0}, {'cbd': 'Pasaraya Manggarai', 'lat': -6.203191, 'lon': 106.843015, 'radius_km': 0.8}, {'cbd': 'pasific place mall', 'lat': -6.241325, 'lon': 106.818512, 'radius_km': 3.4}, {'cbd': 'Pejanten Mall dan Plaza', 'lat': -6.267594, 'lon': 106.829376, 'radius_km': 1.77}, {'cbd': 'Pejaten Mal dan Plaza', 'lat': -6.286338, 'lon': 106.834469, 'radius_km': 1.95}, {'cbd': 'Pejaten Mall & Plaza', 'lat': -6.270361, 'lon': 106.812355, 'radius_km': 3.39}, {'cbd': 'pejaten park', 'lat': -6.28868, 'lon': 106.839308, 'radius_km': 1.8}, {'cbd': 'Pejaten Village', 'lat': -6.269545, 'lon': 106.830718, 'radius_km': 2.44}, {'cbd': 'pejaten village mall', 'lat': -6.269015, 'lon': 106.830057, 'radius_km': 2.04}, {'cbd': 'Pertokoan Pasar Baru', 'lat': -6.163929, 'lon': 106.833893, 'radius_km': 1.0}, {'cbd': 'PGC', 'lat': -6.287065, 'lon': 106.85701, 'radius_km': 4.4}, {'cbd': 'PGC Cililitan', 'lat': -6.262635, 'lon': 106.857309, 'radius_km': 3.28}, {'cbd': 'PIK Avenue', 'lat': -6.116981, 'lon': 106.745669, 'radius_km': 3.08}, {'cbd': 'PIK Mall Avenue', 'lat': -6.118749, 'lon': 106.747566, 'radius_km': 1.0}, {'cbd': 'PIM', 'lat': -6.249612, 'lon': 106.777845, 'radius_km': 2.0}, {'cbd': 'PIM 1', 'lat': -6.2301, 'lon': 106.785144, 'radius_km': 4.67}, {'cbd': 'plasa cibubur', 'lat': -6.342041, 'lon': 106.902764, 'radius_km': 6.0}, {'cbd': 'plasa senayan', 'lat': -6.237824, 'lon': 106.795299, 'radius_km': 2.9}, {'cbd': 'Plaza Atrium', 'lat': -6.173577, 'lon': 106.842381, 'radius_km': 2.2}, {'cbd': 'PLAZA BINTARO', 'lat': -6.270004, 'lon': 106.758748, 'radius_km': 2.36}, {'cbd': 'Plaza Bintaro Satoe', 'lat': -6.263187, 'lon': 106.758693, 'radius_km': 1.0}, {'cbd': 'PLAZA BLOK M', 'lat': -6.253068, 'lon': 106.794815, 'radius_km': 1.46}, {'cbd': 'Plaza Festival Mall Kuningan', 'lat': -6.214657, 'lon': 106.833406, 'radius_km': 1.9}, {'cbd': 'plaza indonesia', 'lat': -6.187329, 'lon': 106.835571, 'radius_km': 3.5}, {'cbd': 'Plaza Kenari Mas', 'lat': -6.190795, 'lon': 106.850901, 'radius_km': 1.75}, {'cbd': 'Plaza Kuningan', 'lat': -6.223933, 'lon': 106.832873, 'radius_km': 3.53}, {'cbd': 'plaza pondok gede', 'lat': -6.295293, 'lon': 106.908888, 'radius_km': 2.0}, {'cbd': 'Plaza PP', 'lat': -6.302481, 'lon': 106.854854, 'radius_km': 3.0}, {'cbd': 'Plaza Senayan', 'lat': -6.209731, 'lon': 106.770651, 'radius_km': 4.78}, {'cbd': 'Plaza Slipi', 'lat': -6.192773, 'lon': 106.788891, 'radius_km': 3.2}, {'cbd': 'Plaza slipi jaya', 'lat': -6.189299, 'lon': 106.787841, 'radius_km': 1.06}, {'cbd': 'Pluit Junction Mall', 'lat': -6.125013, 'lon': 106.792533, 'radius_km': 0.5}, {'cbd': 'Pluit village', 'lat': -6.113021, 'lon': 106.788521, 'radius_km': 1.8}, {'cbd': 'Pluit Village Mall', 'lat': -6.097963, 'lon': 106.792962, 'radius_km': 2.07}, {'cbd': 'poin square', 'lat': -6.291414, 'lon': 106.770407, 'radius_km': 3.28}, {'cbd': 'POINS MALL', 'lat': -6.300369, 'lon': 106.774845, 'radius_km': 2.52}, {'cbd': 'poins square', 'lat': -6.295169, 'lon': 106.77164, 'radius_km': 3.0}, {'cbd': 'Point Square', 'lat': -6.292735, 'lon': 106.779927, 'radius_km': 3.75}, {'cbd': 'Pondok Gede Plaza', 'lat': -6.292257, 'lon': 106.909526, 'radius_km': 2.0}, {'cbd': 'Pondok Indah', 'lat': -6.267213, 'lon': 106.787859, 'radius_km': 3.59}, {'cbd': 'Pondok Indah CBD', 'lat': -6.274951, 'lon': 106.769936, 'radius_km': 3.0}, {'cbd': 'pondok indah mall', 'lat': -6.272458, 'lon': 106.774844, 'radius_km': 3.68}, {'cbd': 'pondok indah mall 1', 'lat': -6.259125, 'lon': 106.778765, 'radius_km': 2.95}, {'cbd': 'pondok indah mall 2', 'lat': -6.26781, 'lon': 106.782826, 'radius_km': 0.5}, {'cbd': 'Pondok Indah Mall 3', 'lat': -6.268977, 'lon': 106.771865, 'radius_km': 1.6}, {'cbd': 'Pondok Kelapa Town Square', 'lat': -6.238667, 'lon': 106.922951, 'radius_km': 4.0}, {'cbd': 'PRJ Kemayoran', 'lat': -6.141391, 'lon': 106.856014, 'radius_km': 1.9}, {'cbd': 'PTC Pulo Gadung', 'lat': -6.190168, 'lon': 106.919339, 'radius_km': 3.67}, {'cbd': 'Pulogadung Trade Center', 'lat': -6.187467, 'lon': 106.922371, 'radius_km': 1.9}, {'cbd': 'Pulogadung Trade Centre', 'lat': -6.18014, 'lon': 106.91375, 'radius_km': 1.0}, {'cbd': 'Puri Indah Mall', 'lat': -6.206533, 'lon': 106.729851, 'radius_km': 3.5}, {'cbd': 'Pusat Grosir Metro Cipulir', 'lat': -6.232642, 'lon': 106.760738, 'radius_km': 1.0}, {'cbd': 'Pusat Grosir Perniagaan Pasar jaya', 'lat': -6.14465, 'lon': 106.81057, 'radius_km': 0.5}, {'cbd': 'Ramayana Cengkareng', 'lat': -6.154873, 'lon': 106.744565, 'radius_km': 1.8}, {'cbd': 'ramayana pasar cengkareng', 'lat': -6.147716, 'lon': 106.728864, 'radius_km': 1.0}, {'cbd': 'Ramayana Pasar Kebayoran Lama', 'lat': -6.229836, 'lon': 106.779469, 'radius_km': 1.0}, {'cbd': 'Ramayana Pasar Kopro', 'lat': -6.176462, 'lon': 106.780209, 'radius_km': 1.0}, {'cbd': 'Ramayana Pondok Gede', 'lat': -6.298523, 'lon': 106.906058, 'radius_km': 3.3}, {'cbd': 'Ramayana PTC Pulogadung', 'lat': -6.177342, 'lon': 106.914985, 'radius_km': 2.0}, {'cbd': 'Ramayana Semper', 'lat': -6.129586, 'lon': 106.91925, 'radius_km': 2.6}, {'cbd': 'Ramayana Semper Koja', 'lat': -6.128297, 'lon': 106.920763, 'radius_km': 0.5}, {'cbd': 'Ranch Market', 'lat': -6.188949, 'lon': 106.756526, 'radius_km': 1.0}, {'cbd': 'ratu plaza', 'lat': -6.227278, 'lon': 106.802363, 'radius_km': 0.5}, {'cbd': 'Rawamangun Square', 'lat': -6.199237, 'lon': 106.886674, 'radius_km': 1.59}, {'cbd': 'Rindam Jaya', 'lat': -6.294647, 'lon': 106.855881, 'radius_km': 3.0}, {'cbd': 'Roxy Square', 'lat': -6.173127, 'lon': 106.801053, 'radius_km': 0.95}, {'cbd': 'Ruko Bandengan Megah', 'lat': -6.134158, 'lon': 106.795521, 'radius_km': 0.5}, {'cbd': 'Ruko Green Garden', 'lat': -6.174949, 'lon': 106.758754, 'radius_km': 2.4}, {'cbd': 'ruko puri botanical', 'lat': -6.21957, 'lon': 106.733449, 'radius_km': 2.35}, {'cbd': 'RUKO PURI BOTANIKA', 'lat': -6.217703, 'lon': 106.744179, 'radius_km': 1.0}, {'cbd': 'Ruko Puri Delta Mas', 'lat': -6.139025, 'lon': 106.799173, 'radius_km': 1.0}, {'cbd': 'Rumah Sakit Harum Sisma Medika', 'lat': -6.242072, 'lon': 106.920964, 'radius_km': 2.92}, {'cbd': 'Rumah Sakit Islam Jakarta Sukapura', 'lat': -6.139376, 'lon': 106.917586, 'radius_km': 1.0}, {'cbd': 'Rumah Sakit St. Carolus', 'lat': -6.195638, 'lon': 106.853529, 'radius_km': 1.0}, {'cbd': 'Sarinah', 'lat': -6.185824, 'lon': 106.825132, 'radius_km': 2.8}, {'cbd': 'Sarinah Mall', 'lat': -6.187235, 'lon': 106.825124, 'radius_km': 0.78}, {'cbd': 'SCBD', 'lat': -6.194694, 'lon': 106.88382, 'radius_km': 6.0}, {'cbd': 'Season City', 'lat': -6.159534, 'lon': 106.791878, 'radius_km': 0.8}, {'cbd': 'Seasons City', 'lat': -6.147959, 'lon': 106.774029, 'radius_km': 2.5}, {'cbd': 'sedayu square', 'lat': -6.13124, 'lon': 106.746659, 'radius_km': 3.0}, {'cbd': 'Sekolah NJIS', 'lat': -6.156193, 'lon': 106.896796, 'radius_km': 1.0}, {'cbd': 'senayan', 'lat': -6.222489, 'lon': 106.781752, 'radius_km': 3.22}, {'cbd': 'Senayan City', 'lat': -6.228079, 'lon': 106.790048, 'radius_km': 2.92}, {'cbd': 'senayan city mall', 'lat': -6.229842, 'lon': 106.797178, 'radius_km': 1.0}, {'cbd': 'Senayan Park', 'lat': -6.208916, 'lon': 106.806324, 'radius_km': 1.08}, {'cbd': 'Senen Jaya', 'lat': -6.167096, 'lon': 106.844201, 'radius_km': 2.57}, {'cbd': 'Serengseng Junction', 'lat': -6.206513, 'lon': 106.758349, 'radius_km': 2.04}, {'cbd': 'Slipi Jaya', 'lat': -6.18845, 'lon': 106.792412, 'radius_km': 1.0}, {'cbd': 'Slipi Jaya Plaza', 'lat': -6.206238, 'lon': 106.796241, 'radius_km': 1.9}, {'cbd': 'Srengseng Junction', 'lat': -6.212933, 'lon': 106.759264, 'radius_km': 2.0}, {'cbd': 'Stasiun Bojong Indah', 'lat': -6.161016, 'lon': 106.73446, 'radius_km': 1.0}, {'cbd': 'Stasiun Cikini', 'lat': -6.195198, 'lon': 106.841363, 'radius_km': 2.7}, {'cbd': 'Stasiun Gondangdia', 'lat': -6.190258, 'lon': 106.836137, 'radius_km': 1.5}, {'cbd': 'Stasiun Jatinegara', 'lat': -6.221388, 'lon': 106.868721, 'radius_km': 3.0}, {'cbd': 'stasiun kalideres', 'lat': -6.159225, 'lon': 106.692338, 'radius_km': 2.0}, {'cbd': 'Stasiun Kebayoran', 'lat': -6.239672, 'lon': 106.781035, 'radius_km': 1.0}, {'cbd': 'stasiun kereta cikini', 'lat': -6.200866, 'lon': 106.840844, 'radius_km': 1.0}, {'cbd': 'stasiun lenteng agung', 'lat': -6.335799, 'lon': 106.83563, 'radius_km': 1.0}, {'cbd': 'Stasiun Manggarai', 'lat': -6.204754, 'lon': 106.836019, 'radius_km': 2.0}, {'cbd': 'Stasiun MRT Cipete', 'lat': -6.277834, 'lon': 106.806158, 'radius_km': 1.0}, {'cbd': 'stasiun mrt lebak bulu', 'lat': -6.291586, 'lon': 106.771369, 'radius_km': 0.95}, {'cbd': 'Stasiun Tanah Abanh', 'lat': -6.173542, 'lon': 106.809273, 'radius_km': 5.0}, {'cbd': 'STIKIM', 'lat': -6.330391, 'lon': 106.834181, 'radius_km': 1.01}, {'cbd': 'Sudirman CBD', 'lat': -6.232106, 'lon': 106.792196, 'radius_km': 1.0}, {'cbd': 'Sumareccon Mall Kelapa Gading', 'lat': -6.15446, 'lon': 106.91423, 'radius_km': 1.0}, {'cbd': 'sumarecon mall kelapa gading', 'lat': -6.149216, 'lon': 106.915268, 'radius_km': 2.0}, {'cbd': 'Summarecon Mall', 'lat': -6.1757, 'lon': 106.89762, 'radius_km': 2.0}, {'cbd': 'Summarecon Mall Kelapa Gading', 'lat': -6.164357, 'lon': 106.910509, 'radius_km': 3.15}, {'cbd': 'SUNSET AVENUE CITRA 8', 'lat': -6.121785, 'lon': 106.698264, 'radius_km': 2.95}, {'cbd': 'Sunter Mall', 'lat': -6.137755, 'lon': 106.860896, 'radius_km': 2.97}, {'cbd': 'Sunter Podomoro', 'lat': -6.131896, 'lon': 106.869527, 'radius_km': 1.0}, {'cbd': 'Super Indo Cipinang', 'lat': -6.219939, 'lon': 106.886813, 'radius_km': 1.1}, {'cbd': 'Superindo Bintaro', 'lat': -6.271398, 'lon': 106.76315, 'radius_km': 1.73}, {'cbd': 'SUPERINDO CILEDUK', 'lat': -6.234731, 'lon': 106.744545, 'radius_km': 1.8}, {'cbd': 'taman anggrek mall', 'lat': -6.194713, 'lon': 106.769129, 'radius_km': 3.0}, {'cbd': 'taman impian jaya Ancol', 'lat': -6.122705, 'lon': 106.824584, 'radius_km': 1.0}, {'cbd': 'Taman Menteng', 'lat': -6.198646, 'lon': 106.83563, 'radius_km': 1.1}, {'cbd': 'Tamini Square', 'lat': -6.287706, 'lon': 106.8794, 'radius_km': 3.72}, {'cbd': 'Tanah Impian Jaya Ancol', 'lat': -6.125382, 'lon': 106.84967, 'radius_km': 1.0}, {'cbd': 'TEBET ECO PARK', 'lat': -6.236045, 'lon': 106.852633, 'radius_km': 0.5}, {'cbd': 'TERMINAL BUS TANJUNG PRIOK', 'lat': -6.111749, 'lon': 106.882092, 'radius_km': 1.1}, {'cbd': 'Terminal Kampung Rambutan', 'lat': -6.299864, 'lon': 106.877694, 'radius_km': 4.2}, {'cbd': 'Terminal Pulogebang', 'lat': -6.210414, 'lon': 106.949161, 'radius_km': 1.0}, {'cbd': 'Thamrin City', 'lat': -6.198176, 'lon': 106.81529, 'radius_km': 1.9}, {'cbd': 'The Centro - Metro PIK', 'lat': -6.10819, 'lon': 106.75942, 'radius_km': 1.0}, {'cbd': 'the east tower', 'lat': -6.23046, 'lon': 106.82513, 'radius_km': 1.0}, {'cbd': 'The Park Pejaten', 'lat': -6.273483, 'lon': 106.831446, 'radius_km': 2.52}, {'cbd': 'THE PRIME OFFICE SUITE', 'lat': -6.12919, 'lon': 106.893398, 'radius_km': 1.0}, {'cbd': 'Tip Top Pondok Bambu', 'lat': -6.248909, 'lon': 106.904966, 'radius_km': 2.56}, {'cbd': 'tip top rawamangun', 'lat': -6.202314, 'lon': 106.896903, 'radius_km': 1.2}, {'cbd': 'TMII', 'lat': -6.307693, 'lon': 106.902416, 'radius_km': 6.0}, {'cbd': 'Tower Cibis CBD', 'lat': -6.310341, 'lon': 106.826917, 'radius_km': 2.77}, {'cbd': 'Tower Cibis Park CBD', 'lat': -6.306037, 'lon': 106.824062, 'radius_km': 2.31}, {'cbd': 'Trans Studio Mall Cibubur', 'lat': -6.361623, 'lon': 106.89898, 'radius_km': 2.37}, {'cbd': 'Transamart Cilandak', 'lat': -6.314111, 'lon': 106.806367, 'radius_km': 2.0}, {'cbd': 'transmart carefour cempaka putih', 'lat': -6.175173, 'lon': 106.876069, 'radius_km': 1.0}, {'cbd': 'Transmart Carrefour Cempaka Putih', 'lat': -6.173363, 'lon': 106.880072, 'radius_km': 1.07}, {'cbd': 'Transmart Cempaka Putih', 'lat': -6.172978, 'lon': 106.872226, 'radius_km': 3.26}, {'cbd': 'Transmart Cilandak', 'lat': -6.310964, 'lon': 106.810927, 'radius_km': 4.12}, {'cbd': 'Transmart Kalimalang', 'lat': -6.242432, 'lon': 106.92063, 'radius_km': 2.0}, {'cbd': 'Universitas Negeri Jakarta', 'lat': -6.194766, 'lon': 106.883722, 'radius_km': 1.84}, {'cbd': 'Universitas Pancasila', 'lat': -6.337265, 'lon': 106.835854, 'radius_km': 1.0}, {'cbd': 'Universitas Pertamina', 'lat': -6.226021, 'lon': 106.785809, 'radius_km': 1.0}, {'cbd': 'Wisata Kota Tua', 'lat': -6.136366, 'lon': 106.806, 'radius_km': 0.8}, {'cbd': 'Wisata Taman Impian Jaya Ancol', 'lat': -6.130843, 'lon': 106.82473, 'radius_km': 1.0}, {'cbd': 'WTC mangga dua', 'lat': -6.124006, 'lon': 106.82481, 'radius_km': 2.0}]


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
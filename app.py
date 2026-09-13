
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import os
import math
import re
import requests
from rapidfuzz import fuzz, process as rprocess
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='MediScan AI', page_icon='🩺', layout='wide')

CSS = """
<style>
  section[data-testid="stSidebar"]{display:none}
  .stButton>button{border-radius:10px}
  .about-card{background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#f0f9ff 100%);
    border:1px solid #e2e8f0;border-radius:16px;padding:28px 32px;
    margin-bottom:24px;box-shadow:0 1px 3px rgba(15,23,42,0.04)}
  .about-title{font-size:24px;font-weight:700;margin-bottom:6px;color:#1e293b}
  .about-title .accent{background:linear-gradient(90deg,#6366f1,#3b82f6,#10b981);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
  .about-sub{font-size:13px;color:#64748b;margin-bottom:20px}
  .stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
  .stat-box{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:14px;
    text-align:center;box-shadow:0 1px 2px rgba(15,23,42,0.03)}
  .stat-num{font-size:22px;font-weight:700;color:#6366f1}
  .stat-label{font-size:11px;color:#64748b;margin-top:2px}
  .about-para{font-size:13px;color:#475569;line-height:1.7;margin:14px 0}
  .feature-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:16px}
  .feat-box{background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
    padding:12px 14px;box-shadow:0 1px 2px rgba(15,23,42,0.03)}
  .feat-icon{font-size:18px;margin-bottom:4px}
  .feat-title{font-size:12px;font-weight:600;color:#1e293b}
  .feat-desc{font-size:11px;color:#64748b;margin-top:2px;line-height:1.5}
  .rf-badge{background:#EEEDFE;color:#3C3489;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600}
  .lstm-badge{background:#E1F5EE;color:#0F6E56;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600}
  .pred-card{background:#f8f9fa;border-radius:12px;padding:16px;margin:8px 0;border-left:4px solid #534AB7}
  .conf-bar{height:8px;background:#e9ecef;border-radius:4px;overflow:hidden;margin-top:6px}
  .conf-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#534AB7,#7C3AED)}
  .sym-tag{display:inline-block;background:#EEEDFE;color:#3C3489;padding:3px 10px;
    border-radius:20px;font-size:11px;margin:2px 3px;border:1px solid #C4BBF5}
  .sym-miss{display:inline-block;background:#FAEEDA;color:#633806;padding:3px 10px;
    border-radius:20px;font-size:11px;margin:2px 3px;border:1px solid #F0C070}
  .place-card{background:#f8f9fa;border-radius:10px;padding:12px;margin-bottom:8px;border:1px solid #e9ecef}
  .section-card{background:#fafbfc;border:1px solid #e9ecef;border-radius:12px;padding:16px 20px;margin-bottom:12px}
  .section-label{font-size:13px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
  .section-desc{font-size:12px;color:#64748b;margin-bottom:12px;line-height:1.5}
  .key-sym-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
    padding:14px;text-align:center;box-shadow:0 1px 2px rgba(15,23,42,0.04)}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

for k, v in {'history': [], 'show_results': False, 'selected_region': 'all',
             'reset_counter': 0, 'input_mode': 'text'}.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, 'models')
    data_dir = os.path.join(base, 'data')

    # ── Load COMPRESSED RF model ──────────────────────────
    rf_model = joblib.load(os.path.join(models_dir, 'rf_model_compressed.joblib'))
    rf_le = pickle.load(open(os.path.join(models_dir, 'rf_label_encoder.pkl'), 'rb'))
    rf_symptoms = pickle.load(open(os.path.join(models_dir, 'rf_symptom_list.pkl'), 'rb'))

    # ── Load LSTM model ───────────────────────────────────
    lstm_model = None; lstm_le = lstm_sym_to_idx = lstm_max_len = None
    try:
        import tensorflow as tf
        lstm_model = tf.keras.models.load_model(os.path.join(models_dir, 'lstm_model.keras'))
        lstm_le = pickle.load(open(os.path.join(models_dir, 'lstm_label_encoder.pkl'), 'rb'))
        lstm_sym_to_idx = pickle.load(open(os.path.join(models_dir, 'lstm_symptom_to_idx.pkl'), 'rb'))
        lstm_max_len = pickle.load(open(os.path.join(models_dir, 'lstm_max_len.pkl'), 'rb'))
    except Exception as e:
        print(f"LSTM not loaded: {e}")

    # ── Load metrics ──────────────────────────────────────
    rf_metrics = {'top1': 0, 'top3': 0, 'top5': 0, 'cal_gap': None}
    lstm_metrics = {'top1': 0, 'top3': 0, 'top5': 0}
    try:
        with open(os.path.join(data_dir, 'model_comparison.json')) as f:
            comp = json.load(f)['models']
        rf_metrics = {k: comp['RandomForest'].get(k, 0) for k in ['top1','top3','top5']}
        rf_metrics['cal_gap'] = comp['RandomForest'].get('cal_gap')
        lstm_metrics = {k: comp['LSTM'].get(k, 0) for k in ['top1','top3','top5']}
    except Exception as e:
        print(f"comparison not found: {e}")

    return {
        'rf': {'model': rf_model, 'le': rf_le, 'symptoms': rf_symptoms, 'metrics': rf_metrics},
        'lstm': {'model': lstm_model, 'le': lstm_le, 'sym_to_idx': lstm_sym_to_idx,
                 'max_len': lstm_max_len, 'metrics': lstm_metrics},
    }

try:
    M = load_models()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

SPECIALIST_MAP = {
    'heart':'Cardiologist','cardiac':'Cardiologist','angina':'Cardiologist',
    'lung':'Pulmonologist','asthma':'Pulmonologist','pneumonia':'Pulmonologist',
    'diabetes':'Endocrinologist','thyroid':'Endocrinologist',
    'cancer':'Oncologist','tumor':'Oncologist',
    'skin':'Dermatologist','acne':'Dermatologist','psoriasis':'Dermatologist',
    'brain':'Neurologist','migraine':'Neurologist','epilepsy':'Neurologist','stroke':'Neurologist',
    'anxiety':'Psychiatrist','depression':'Psychiatrist',
    'kidney':'Nephrologist','urinary':'Urologist','bladder':'Urologist',
    'eye':'Ophthalmologist','ear':'ENT Specialist',
    'bone':'Orthopedist','arthritis':'Rheumatologist','joint':'Orthopedist',
    'liver':'Gastroenterologist','hepatitis':'Hepatologist',
    'malaria':'Infectious Disease Specialist','dengue':'Infectious Disease Specialist',
}
def get_specialist(d):
    d = d.lower()
    for k, v in SPECIALIST_MAP.items():
        if k in d: return v
    return 'General Physician'

REGION_SYMPTOMS = {
    'head':    ['headache','frontal headache','dizziness','fainting','slurring words',
                'diminished vision','double vision','spots or clouds in vision','facial pain'],
    'chest':   ['sharp chest pain','chest tightness','burning chest pain','shortness of breath',
                'difficulty breathing','cough','wheezing','palpitations','irregular heartbeat',
                'increased heart rate','decreased heart rate'],
    'abdomen': ['sharp abdominal pain','burning abdominal pain','lower abdominal pain',
                'upper abdominal pain','nausea','vomiting','diarrhea','constipation',
                'heartburn','flatulence','stomach bloating','swollen abdomen',
                'decreased appetite','vomiting blood'],
    'skin':    ['skin rash','itching of skin','skin lesion','skin growth','skin moles',
                'skin swelling','skin irritation','abnormal appearing skin','acne or pimples',
                'skin dryness, peeling, scaliness, or roughness','warts','skin pain'],
    'joints':  ['joint pain','joint swelling','joint stiffness or tightness','knee pain',
                'knee swelling','hip pain','shoulder pain','back pain','low back pain',
                'neck pain','muscle pain','muscle weakness','cramps and spasms','bones are painful'],
    'general': ['fever','fatigue','weakness','chills','sweating','feeling ill','ache all over',
                'weight gain','recent weight loss','swollen lymph nodes','allergic reaction',
                'flu-like syndrome','jaundice'],
    'urinary': ['painful urination','frequent urination','blood in urine','retention of urine',
                'low urine output','involuntary urination','unusual color or odor to urine',
                'excessive urination at night','hesitancy','symptoms of bladder'],
    'throat':  ['sore throat','throat swelling','lump in throat','throat feels tight',
                'drainage in throat','hoarse voice','nasal congestion','sinus congestion',
                'coryza','sneezing','nosebleed','painful sinuses'],
}
REGION_META = {
    'all':     ('All',     '#F1F5F9', '#94A3B8', '#334155'),
    'head':    ('Head',    '#EEEDFE', '#534AB7', '#3C3489'),
    'chest':   ('Chest',   '#E1F5EE', '#0F6E56', '#085041'),
    'abdomen': ('Abdomen', '#FAEEDA', '#BA7517', '#633806'),
    'skin':    ('Skin',    '#FBEAF0', '#993556', '#72243E'),
    'joints':  ('Joints',  '#F1EFE8', '#5F5E5A', '#444441'),
    'general': ('General', '#E6F1FB', '#185FA5', '#0C447C'),
    'urinary': ('Urinary', '#E6F1FB', '#185FA5', '#0C447C'),
    'throat':  ('Throat',  '#EEEDFE', '#534AB7', '#3C3489'),
}

SYNONYMS = {
    'fever': 'fever', 'high fever': 'fever', 'temperature': 'fever',
    'pyrexia': 'fever', 'hot body': 'fever', 'feverish': 'fever',
    'bukhar': 'fever', 'बुखार': 'fever', 'बुख़ार': 'fever',
    'jvar': 'fever', 'ज्वर': 'fever', 'bhukhar': 'fever', 'fvr': 'fever',
    'headache': 'headache', 'head pain': 'headache', 'head ache': 'headache',
    'frontal headache': 'frontal headache', 'forehead pain': 'frontal headache',
    'migraine': 'headache',
    'sar dard': 'headache', 'सर दर्द': 'headache', 'सिर दर्द': 'headache',
    'sir dard': 'headache', 'sar me dard': 'headache',
    'cough': 'cough', 'coughing': 'cough', 'dry cough': 'cough',
    'wet cough': 'cough', 'khansi': 'cough', 'खांसी': 'cough', 'खाँसी': 'cough',
    'khaansi': 'cough', 'cogh': 'cough',
    'coughing up sputum': 'coughing up sputum', 'cough with sputum': 'coughing up sputum',
    'phlegm': 'coughing up sputum', 'mucus cough': 'coughing up sputum',
    'balgam': 'coughing up sputum', 'बलगम': 'coughing up sputum',
    'hemoptysis': 'hemoptysis', 'blood in cough': 'hemoptysis', 'coughing blood': 'hemoptysis',
    'shortness of breath': 'shortness of breath', 'short of breath': 'shortness of breath',
    'sob': 'shortness of breath', 'breathless': 'shortness of breath',
    'breathlessness': 'shortness of breath', 'cant breathe': 'shortness of breath',
    'difficulty breathing': 'difficulty breathing', 'breathing problem': 'difficulty breathing',
    'hard to breathe': 'difficulty breathing', 'dyspnea': 'difficulty breathing',
    'saans ki takleef': 'difficulty breathing', 'सांस की तकलीफ': 'difficulty breathing',
    'breathing fast': 'breathing fast', 'rapid breathing': 'breathing fast',
    'hurts to breath': 'hurts to breath', 'painful breathing': 'hurts to breath',
    'wheezing': 'wheezing', 'wheeze': 'wheezing', 'whistling breath': 'wheezing',
    'apnea': 'apnea', 'stopped breathing': 'apnea',
    'sharp chest pain': 'sharp chest pain', 'chest pain': 'sharp chest pain',
    'cp': 'sharp chest pain', 'chest ache': 'sharp chest pain',
    'chest tightness': 'chest tightness', 'tight chest': 'chest tightness',
    'chest pressure': 'chest tightness',
    'burning chest pain': 'burning chest pain', 'heartburn pain': 'burning chest pain',
    'congestion in chest': 'congestion in chest', 'chest congestion': 'congestion in chest',
    'palpitations': 'palpitations', 'heart racing': 'palpitations',
    'heart pounding': 'palpitations', 'dhak dhak': 'palpitations',
    'धड़कन': 'palpitations', 'dhadkan': 'palpitations',
    'irregular heartbeat': 'irregular heartbeat', 'arrhythmia': 'irregular heartbeat',
    'fast heartbeat': 'increased heart rate', 'rapid heartbeat': 'increased heart rate',
    'tachycardia': 'increased heart rate', 'increased heart rate': 'increased heart rate',
    'slow heartbeat': 'decreased heart rate', 'bradycardia': 'decreased heart rate',
    'decreased heart rate': 'decreased heart rate',
    'sharp abdominal pain': 'sharp abdominal pain', 'abdominal pain': 'sharp abdominal pain',
    'stomach pain': 'sharp abdominal pain', 'stomachache': 'sharp abdominal pain',
    'stomach ache': 'sharp abdominal pain', 'tummy ache': 'sharp abdominal pain',
    'tummy pain': 'sharp abdominal pain', 'pet dard': 'sharp abdominal pain',
    'पेट दर्द': 'sharp abdominal pain', 'pet me dard': 'sharp abdominal pain',
    'burning abdominal pain': 'burning abdominal pain', 'burning stomach': 'burning abdominal pain',
    'lower abdominal pain': 'lower abdominal pain', 'lower stomach pain': 'lower abdominal pain',
    'upper abdominal pain': 'upper abdominal pain', 'upper stomach pain': 'upper abdominal pain',
    'suprapubic pain': 'suprapubic pain', 'below navel pain': 'suprapubic pain',
    'abdominal distention': 'abdominal distention', 'bloated stomach': 'abdominal distention',
    'bloating': 'stomach bloating', 'stomach bloating': 'stomach bloating',
    'swollen abdomen': 'swollen abdomen', 'bloated belly': 'swollen abdomen',
    'side pain': 'side pain', 'flank pain': 'side pain', 'groin pain': 'groin pain',
    'nausea': 'nausea', 'nauseous': 'nausea', 'feel like vomiting': 'nausea',
    'ji michalna': 'nausea', 'जी मिचलाना': 'nausea',
    'vomiting': 'vomiting', 'vomit': 'vomiting', 'throwing up': 'vomiting',
    'puking': 'vomiting', 'puke': 'vomiting', 'ulti': 'vomiting',
    'उल्टी': 'vomiting', 'उलटी': 'vomiting',
    'vomiting blood': 'vomiting blood', 'blood in vomit': 'vomiting blood',
    'hematemesis': 'vomiting blood',
    'regurgitation': 'regurgitation', 'food coming back': 'regurgitation',
    'heartburn': 'heartburn', 'acid reflux': 'heartburn', 'acidity': 'heartburn',
    'indigestion': 'heartburn', 'gas': 'flatulence', 'flatulence': 'flatulence',
    'gassy': 'flatulence', 'apach': 'heartburn', 'अपच': 'heartburn',
    'constipation': 'constipation', 'constipated': 'constipation',
    'kabz': 'constipation', 'कब्ज': 'constipation',
    'diarrhea': 'diarrhea', 'diarrhoea': 'diarrhea', 'loose motion': 'diarrhea',
    'loose motions': 'diarrhea', 'loose stool': 'diarrhea', 'watery stool': 'diarrhea',
    'dast': 'diarrhea', 'दस्त': 'diarrhea',
    'blood in stool': 'blood in stool', 'bloody stool': 'blood in stool',
    'rectal bleeding': 'rectal bleeding', 'bleeding from anus': 'rectal bleeding',
    'melena': 'melena', 'black stool': 'melena', 'dark stool': 'melena',
    'decreased appetite': 'decreased appetite', 'loss of appetite': 'decreased appetite',
    'no appetite': 'decreased appetite', 'bhookh nahi': 'decreased appetite',
    'भूख नहीं': 'decreased appetite', 'bhookh kam': 'decreased appetite',
    'excessive appetite': 'excessive appetite', 'always hungry': 'excessive appetite',
    'skin rash': 'skin rash', 'rash': 'skin rash', 'redness on skin': 'skin rash',
    'itching of skin': 'itching of skin', 'itchy skin': 'itching of skin',
    'itching': 'itching of skin', 'itch': 'itching of skin',
    'khujli': 'itching of skin', 'खुजली': 'itching of skin',
    'skin lesion': 'skin lesion', 'skin sore': 'skin lesion',
    'skin growth': 'skin growth', 'skin lump': 'skin growth',
    'skin moles': 'skin moles', 'mole': 'skin moles', 'moles': 'skin moles',
    'skin swelling': 'skin swelling', 'swollen skin': 'skin swelling',
    'skin irritation': 'skin irritation', 'skin pain': 'skin pain',
    'abnormal appearing skin': 'abnormal appearing skin',
    'skin dryness': 'skin dryness, peeling, scaliness, or roughness',
    'dry skin': 'skin dryness, peeling, scaliness, or roughness',
    'flaky skin': 'skin dryness, peeling, scaliness, or roughness',
    'peeling skin': 'skin dryness, peeling, scaliness, or roughness',
    'scaly skin': 'skin dryness, peeling, scaliness, or roughness',
    'wrinkles on skin': 'wrinkles on skin', 'wrinkles': 'wrinkles on skin',
    'acne or pimples': 'acne or pimples', 'acne': 'acne or pimples',
    'pimples': 'acne or pimples', 'pimple': 'acne or pimples',
    'diaper rash': 'diaper rash', 'warts': 'warts',
    'dizziness': 'dizziness', 'dizzy': 'dizziness', 'vertigo': 'dizziness',
    'lightheaded': 'dizziness', 'light headed': 'dizziness', 'giddy': 'dizziness',
    'chakkar': 'dizziness', 'चक्कर': 'dizziness', 'sir ghoomna': 'dizziness',
    'fainting': 'fainting', 'syncope': 'fainting', 'passed out': 'fainting',
    'facial pain': 'facial pain', 'face pain': 'facial pain',
    'jaw pain': 'jaw pain', 'jaw swelling': 'jaw swelling',
    'slurring words': 'slurring words', 'slurred speech': 'slurring words',
    'difficulty speaking': 'difficulty speaking', 'cannot speak': 'difficulty speaking',
    'eye pain': 'pain in eye', 'pain in eye': 'pain in eye',
    'eye redness': 'eye redness', 'red eye': 'eye redness',
    'itchiness of eye': 'itchiness of eye', 'itchy eyes': 'itchiness of eye',
    'eye burns or stings': 'eye burns or stings', 'burning eyes': 'eye burns or stings',
    'blurred vision': 'diminished vision', 'blurry vision': 'diminished vision',
    'diminished vision': 'diminished vision', 'vision loss': 'diminished vision',
    'cant see': 'diminished vision', 'dhundhla': 'diminished vision',
    'double vision': 'double vision', 'diplopia': 'double vision',
    'spots or clouds in vision': 'spots or clouds in vision', 'floaters': 'spots or clouds in vision',
    'swollen eye': 'swollen eye', 'puffy eyes': 'swollen eye',
    'eyelid swelling': 'eyelid swelling', 'swollen eyelid': 'eyelid swelling',
    'itchy eyelid': 'itchy eyelid',
    'foreign body sensation in eye': 'foreign body sensation in eye',
    'something in eye': 'foreign body sensation in eye',
    'blindness': 'blindness', 'blind': 'blindness',
    'cross-eyed': 'cross-eyed',
    'bleeding from eye': 'bleeding from eye', 'cloudy eye': 'cloudy eye',
    'lacrimation': 'lacrimation', 'watery eyes': 'lacrimation',
    'ear pain': 'ear pain', 'earache': 'ear pain', 'ear ache': 'ear pain',
    'kan dard': 'ear pain', 'कान दर्द': 'ear pain',
    'ringing in ear': 'ringing in ear', 'tinnitus': 'ringing in ear',
    'diminished hearing': 'diminished hearing', 'hearing loss': 'diminished hearing',
    'hearing problem': 'diminished hearing',
    'fluid in ear': 'fluid in ear',
    'plugged feeling in ear': 'plugged feeling in ear', 'blocked ear': 'plugged feeling in ear',
    'redness in ear': 'redness in ear',
    'pus draining from ear': 'pus draining from ear', 'ear discharge': 'pus draining from ear',
    'bleeding from ear': 'bleeding from ear',
    'pulling at ears': 'pulling at ears', 'ear pulling': 'pulling at ears',
    'itchy ear': 'itchy ear(s)', 'itchy ears': 'itchy ear(s)',
    'nasal congestion': 'nasal congestion', 'blocked nose': 'nasal congestion',
    'stuffy nose': 'nasal congestion', 'nose blocked': 'nasal congestion',
    'naak band': 'nasal congestion', 'नाक बंद': 'nasal congestion',
    'sinus congestion': 'sinus congestion', 'sinus pressure': 'sinus congestion',
    'painful sinuses': 'painful sinuses', 'sinus pain': 'painful sinuses',
    'nosebleed': 'nosebleed', 'nose bleed': 'nosebleed', 'epistaxis': 'nosebleed',
    'sore in nose': 'sore in nose', 'nasal sore': 'sore in nose',
    'redness in or around nose': 'redness in or around nose',
    'coryza': 'coryza', 'runny nose': 'coryza', 'nose running': 'coryza',
    'naak behna': 'coryza', 'नाक बहना': 'coryza',
    'sore throat': 'sore throat', 'throat pain': 'sore throat',
    'gale me dard': 'sore throat', 'गले में दर्द': 'sore throat',
    'gala kharab': 'sore throat', 'throat ache': 'sore throat',
    'throat swelling': 'throat swelling', 'swollen throat': 'throat swelling',
    'throat redness': 'throat redness', 'red throat': 'throat redness',
    'throat irritation': 'throat irritation',
    'throat feels tight': 'throat feels tight', 'tight throat': 'throat feels tight',
    'lump in throat': 'lump in throat', 'something stuck in throat': 'lump in throat',
    'drainage in throat': 'drainage in throat', 'post nasal drip': 'drainage in throat',
    'swollen or red tonsils': 'swollen or red tonsils', 'swollen tonsils': 'swollen or red tonsils',
    'hoarse voice': 'hoarse voice', 'hoarse': 'hoarse voice', 'raspy voice': 'hoarse voice',
    'difficulty in swallowing': 'difficulty in swallowing',
    'trouble swallowing': 'difficulty in swallowing', 'dysphagia': 'difficulty in swallowing',
    'back pain': 'back pain', 'backache': 'back pain', 'back ache': 'back pain',
    'peeth dard': 'back pain', 'पीठ दर्द': 'back pain',
    'low back pain': 'low back pain', 'lower back pain': 'low back pain',
    'back stiffness': 'back stiffness or tightness',
    'back cramps': 'back cramps or spasms', 'back spasms': 'back cramps or spasms',
    'back mass': 'back mass or lump', 'back lump': 'back mass or lump',
    'back swelling': 'back swelling', 'back weakness': 'back weakness',
    'neck pain': 'neck pain', 'gardan dard': 'neck pain', 'गर्दन दर्द': 'neck pain',
    'neck stiffness': 'neck stiffness or tightness',
    'neck cramps': 'neck cramps or spasms', 'neck spasms': 'neck cramps or spasms',
    'neck mass': 'neck mass', 'neck lump': 'neck mass',
    'neck swelling': 'neck swelling', 'neck weakness': 'neck weakness',
    'stiffness all over': 'stiffness all over', 'body stiffness': 'stiffness all over',
    'arm pain': 'arm pain', 'arm ache': 'arm pain',
    'arm swelling': 'arm swelling', 'arm weakness': 'arm weakness',
    'arm stiffness': 'arm stiffness or tightness',
    'hand or finger pain': 'hand or finger pain', 'hand pain': 'hand or finger pain',
    'finger pain': 'hand or finger pain',
    'hand or finger swelling': 'hand or finger swelling',
    'hand or finger weakness': 'hand or finger weakness',
    'wrist pain': 'wrist pain', 'wrist ache': 'wrist pain',
    'wrist swelling': 'wrist swelling', 'wrist weakness': 'wrist weakness',
    'elbow pain': 'elbow pain', 'elbow ache': 'elbow pain',
    'elbow swelling': 'elbow swelling', 'elbow weakness': 'elbow weakness',
    'leg pain': 'leg pain', 'leg ache': 'leg pain', 'pair dard': 'leg pain',
    'पैर दर्द': 'leg pain',
    'leg swelling': 'leg swelling', 'swollen legs': 'leg swelling',
    'leg weakness': 'leg weakness', 'leg stiffness': 'leg stiffness or tightness',
    'leg cramps or spasms': 'leg cramps or spasms', 'leg spasms': 'leg cramps or spasms',
    'foot or toe pain': 'foot or toe pain', 'foot pain': 'foot or toe pain',
    'toe pain': 'foot or toe pain',
    'foot or toe swelling': 'foot or toe swelling',
    'foot or toe weakness': 'foot or toe weakness',
    'ankle pain': 'ankle pain', 'ankle ache': 'ankle pain',
    'ankle swelling': 'ankle swelling', 'ankle weakness': 'ankle weakness',
    'joint pain': 'joint pain', 'joint ache': 'joint pain', 'jodo me dard': 'joint pain',
    'जोड़ों में दर्द': 'joint pain',
    'joint swelling': 'joint swelling', 'swollen joints': 'joint swelling',
    'joint stiffness': 'joint stiffness or tightness',
    'stiff joints': 'joint stiffness or tightness',
    'knee pain': 'knee pain', 'knee ache': 'knee pain', 'ghutne me dard': 'knee pain',
    'घुटनों में दर्द': 'knee pain',
    'knee swelling': 'knee swelling', 'swollen knee': 'knee swelling',
    'knee weakness': 'knee weakness', 'knee stiffness': 'knee stiffness or tightness',
    'hip pain': 'hip pain', 'hip ache': 'hip pain',
    'hip stiffness': 'hip stiffness or tightness',
    'shoulder pain': 'shoulder pain', 'shoulder ache': 'shoulder pain',
    'shoulder swelling': 'shoulder swelling', 'shoulder weakness': 'shoulder weakness',
    'shoulder stiffness': 'shoulder stiffness or tightness',
    'muscle pain': 'muscle pain', 'myalgia': 'muscle pain',
    'muscle weakness': 'muscle weakness', 'muscle swelling': 'muscle swelling',
    'muscle stiffness': 'muscle stiffness or tightness',
    'muscle cramps': 'muscle cramps, contractures, or spasms',
    'cramps and spasms': 'cramps and spasms', 'cramps': 'cramps and spasms',
    'bones are painful': 'bones are painful', 'bone pain': 'bones are painful',
    'rib pain': 'rib pain',
    'painful urination': 'painful urination', 'burning urine': 'painful urination',
    'burning urination': 'painful urination', 'dysuria': 'painful urination',
    'peshab me jalan': 'painful urination', 'पेशाब में जलन': 'painful urination',
    'frequent urination': 'frequent urination', 'urinating often': 'frequent urination',
    'polyuria': 'frequent urination', 'bar bar peshab': 'frequent urination',
    'बार-बार पेशाब': 'frequent urination',
    'excessive urination at night': 'excessive urination at night', 'nocturia': 'excessive urination at night',
    'involuntary urination': 'involuntary urination', 'urine leakage': 'involuntary urination',
    'blood in urine': 'blood in urine', 'hematuria': 'blood in urine',
    'unusual color or odor to urine': 'unusual color or odor to urine',
    'dark urine': 'unusual color or odor to urine', 'smelly urine': 'unusual color or odor to urine',
    'retention of urine': 'retention of urine', 'cannot urinate': 'retention of urine',
    'low urine output': 'low urine output', 'less urine': 'low urine output',
    'hesitancy': 'hesitancy', 'difficulty starting urination': 'hesitancy',
    'anxiety and nervousness': 'anxiety and nervousness', 'anxiety': 'anxiety and nervousness',
    'nervous': 'anxiety and nervousness', 'ghabrahat': 'anxiety and nervousness',
    'घबराहट': 'anxiety and nervousness',
    'depression': 'depression', 'depressed': 'depression', 'udaas': 'depression',
    'उदास': 'depression', 'sad': 'depression',
    'delusions or hallucinations': 'delusions or hallucinations',
    'seeing things': 'delusions or hallucinations', 'hearing voices': 'delusions or hallucinations',
    'disturbance of memory': 'disturbance of memory', 'memory loss': 'disturbance of memory',
    'forgetful': 'disturbance of memory',
    'excessive anger': 'excessive anger', 'anger issues': 'excessive anger',
    'hostile behavior': 'hostile behavior', 'aggressive': 'hostile behavior',
    'temper problems': 'temper problems', 'short temper': 'temper problems',
    'fears and phobias': 'fears and phobias', 'fear': 'fears and phobias',
    'phobia': 'fears and phobias',
    'low self-esteem': 'low self-esteem', 'low confidence': 'low self-esteem',
    'obsessions and compulsions': 'obsessions and compulsions', 'ocd': 'obsessions and compulsions',
    'restlessness': 'restlessness', 'restless': 'restlessness',
    'nightmares': 'nightmares', 'bad dreams': 'nightmares',
    'sleepiness': 'sleepiness', 'sleepy': 'sleepiness', 'drowsy': 'sleepiness',
    'insomnia': 'insomnia', 'cant sleep': 'insomnia', 'neend nahi': 'insomnia',
    'नींद नहीं': 'insomnia', 'sleeplessness': 'insomnia',
    'fatigue': 'fatigue', 'tired': 'fatigue', 'tiredness': 'fatigue',
    'exhausted': 'fatigue', 'thakan': 'fatigue', 'थकान': 'fatigue',
    'weakness': 'weakness', 'weak': 'weakness', 'kamzori': 'weakness',
    'कमजोरी': 'weakness',
    'ache all over': 'ache all over', 'body ache': 'ache all over',
    'body pain': 'ache all over', 'whole body pain': 'ache all over',
    'feeling ill': 'feeling ill', 'unwell': 'feeling ill', 'sick': 'feeling ill',
    'chills': 'chills', 'shivering': 'chills', 'thandi lagna': 'chills',
    'ठंड लगना': 'chills', 'sardi': 'chills', 'सर्दी': 'chills',
    'sweating': 'sweating', 'excessive sweating': 'sweating', 'pasina': 'sweating',
    'पसीना': 'sweating', 'night sweats': 'sweating',
    'hot flashes': 'hot flashes',
    'feeling cold': 'feeling cold', 'always cold': 'feeling cold',
    'feeling hot': 'feeling hot',
    'fluid retention': 'fluid retention', 'water retention': 'fluid retention',
    'peripheral edema': 'peripheral edema', 'swollen ankles': 'peripheral edema',
    'swollen feet': 'peripheral edema',
    'lymphedema': 'lymphedema', 'flushing': 'flushing',
    'jaundice': 'jaundice', 'yellow skin': 'jaundice', 'yellow eyes': 'jaundice',
    'pallor': 'pallor', 'pale skin': 'pallor',
    'thirst': 'thirst', 'excessive thirst': 'thirst',
    'weight gain': 'weight gain', 'wajan badh': 'weight gain', 'वजन बढ़': 'weight gain',
    'recent weight loss': 'recent weight loss', 'weight loss': 'recent weight loss',
    'wajan kam': 'recent weight loss', 'वजन कम': 'recent weight loss',
    'underweight': 'underweight',
    'loss of sensation': 'loss of sensation', 'numbness': 'loss of sensation',
    'paresthesia': 'paresthesia', 'tingling': 'paresthesia', 'pins and needles': 'paresthesia',
    'poor circulation': 'poor circulation',
    'swollen lymph nodes': 'swollen lymph nodes', 'swollen glands': 'swollen lymph nodes',
    'allergic reaction': 'allergic reaction', 'allergy': 'allergic reaction',
    'flu-like syndrome': 'flu-like syndrome', 'flu': 'flu-like syndrome',
    'sneezing': 'sneezing', 'chheenk': 'sneezing', 'छींक': 'sneezing',
    'painful menstruation': 'painful menstruation', 'period pain': 'painful menstruation',
    'menstrual cramps': 'painful menstruation',
    'heavy menstrual flow': 'heavy menstrual flow', 'heavy periods': 'heavy menstrual flow',
    'long menstrual periods': 'long menstrual periods',
    'frequent menstruation': 'frequent menstruation',
    'absence of menstruation': 'absence of menstruation', 'no periods': 'absence of menstruation',
    'irregular periods': 'unpredictable menstruation',
    'unpredictable menstruation': 'unpredictable menstruation',
    'scanty menstrual flow': 'scanty menstrual flow', 'light periods': 'scanty menstrual flow',
    'intermenstrual bleeding': 'intermenstrual bleeding',
    'vaginal discharge': 'vaginal discharge',
    'vaginal itching': 'vaginal itching', 'vaginal itch': 'vaginal itching',
    'vaginal pain': 'vaginal pain',
    'vaginal redness': 'vaginal redness',
    'vaginal dryness': 'vaginal dryness',
    'vulvar irritation': 'vulvar irritation',
    'vulvar sore': 'vulvar sore',
    'pain in testicles': 'pain in testicles', 'testicle pain': 'pain in testicles',
    'swelling of scrotum': 'swelling of scrotum', 'scrotal swelling': 'swelling of scrotum',
    'penile discharge': 'penile discharge',
    'penis pain': 'penis pain', 'penis redness': 'penis redness',
    'bumps on penis': 'bumps on penis',
    'impotence': 'impotence', 'erectile dysfunction': 'impotence',
    'loss of sex drive': 'loss of sex drive', 'low libido': 'loss of sex drive',
    'mouth ulcer': 'mouth ulcer', 'mouth sores': 'mouth ulcer', 'canker sore': 'mouth ulcer',
    'mouth pain': 'mouth pain', 'mouth dryness': 'mouth dryness', 'dry mouth': 'mouth dryness',
    'lip sore': 'lip sore', 'lip swelling': 'lip swelling',
    'gum pain': 'gum pain', 'pain in gums': 'pain in gums',
    'bleeding gums': 'bleeding gums',
    'toothache': 'toothache', 'tooth pain': 'toothache',
    'tongue lesions': 'tongue lesions', 'tongue sores': 'tongue lesions',
    'tongue pain': 'tongue pain',
    'itching of the anus': 'itching of the anus', 'anal itching': 'itching of the anus',
    'pain of the anus': 'pain of the anus', 'anal pain': 'pain of the anus',
    'hemorrhoids': 'mass or swelling around the anus',
    'sob': 'shortness of breath', 'ha': 'headache', 'cp': 'sharp chest pain',
    'doe': 'shortness of breath', 'loc': 'fainting',
    'n/v': 'nausea', 'nvd': 'nausea',
    'uri': 'painful urination', 'uti': 'painful urination',
    'ed': 'impotence', 'msk': 'joint pain', 'od': 'dizziness',
    'fevr': 'fever', 'feverr': 'fever', 'fver': 'fever',
    'hedache': 'headache', 'headach': 'headache',
    'cogh': 'cough', 'coughh': 'cough', 'coff': 'cough',
    'stomache': 'sharp abdominal pain', 'stomack pain': 'sharp abdominal pain',
    'vomitng': 'vomiting', 'vomitting': 'vomiting',
    'dizines': 'dizziness', 'dizzines': 'dizziness', 'dizy': 'dizziness',
    'fatige': 'fatigue', 'fatigu': 'fatigue', 'tird': 'fatigue',
    'brethless': 'shortness of breath', 'breathlesness': 'shortness of breath',
    'diarhea': 'diarrhea', 'diarreah': 'diarrhea',
    'constipaton': 'constipation',
    'sore throt': 'sore throat', 'throt pain': 'sore throat',
    'backpain': 'back pain', 'back pane': 'back pain',
    'chestpain': 'sharp chest pain', 'chest pane': 'sharp chest pain',
}

def match_symptoms(text, symptom_list):
    if not text.strip():
        return [], []
    readable = {s.lower(): s for s in symptom_list}
    readable_list = list(readable.keys())
    STOP = {'a','an','the','and','or','of','in','my','i','have','got','some',
            'feel','feeling','having','with','am','is','are','me','to','for',
            'being','been','was','were','do','does','did','but','if',
            'from','this','that','these','those','will','would','could','should',
            'mild','severe','slight','bit','little','very','so','much','lot'}

    raw_split = re.split(r'[,\n;]|\band\b|\baur\b|&', text.lower())
    terms = [t.strip() for t in raw_split if t.strip()]
    matched, not_found = [], []

    for raw_term in terms:
        term = raw_term.strip().strip('.').strip('!').strip('?').strip()
        if len(term) < 2:
            continue
        found = None

        if term in SYNONYMS:
            c = SYNONYMS[term]
            if c in symptom_list: found = c

        if not found:
            for sev in ['mild','severe','slight','extreme','bad','terrible','high','low','very','chronic','acute']:
                if term.startswith(sev + ' '):
                    term_alt = term[len(sev)+1:].strip()
                    if term_alt in SYNONYMS:
                        c = SYNONYMS[term_alt]
                        if c in symptom_list: found = c; break
                    term = term_alt
                    break

        if not found and term in readable:
            found = readable[term]

        if not found and len(term) >= 4:
            candidates = [s for s in symptom_list if term in s.lower()]
            if candidates:
                found = min(candidates, key=len)

        if not found:
            for s in symptom_list:
                if s.lower() in term: found = s; break

        if not found:
            user_words = set(term.split()) - STOP
            if user_words:
                best_score = 0; best_sym = None
                for s in symptom_list:
                    sym_words = set(s.lower().split()) - STOP
                    if not sym_words: continue
                    overlap = len(user_words & sym_words)
                    if overlap > 0:
                        score = overlap / max(len(sym_words), 1)
                        if score > best_score:
                            best_score = score; best_sym = s
                if best_sym and (best_score >= 0.5 or
                                 len(user_words & set(best_sym.lower().split())) >= 2):
                    found = best_sym

        if not found and ' ' not in term:
            for i in range(3, len(term) - 2):
                combo = term[:i] + ' ' + term[i:]
                if combo in readable: found = readable[combo]; break

        if not found:
            try:
                res = rprocess.extractOne(term, readable_list,
                                          scorer=fuzz.token_set_ratio, score_cutoff=75)
                if res: found = readable[res[0]]
            except: pass

        if not found and len(term) >= 5:
            try:
                res = rprocess.extractOne(term, readable_list,
                                          scorer=fuzz.partial_ratio, score_cutoff=82)
                if res: found = readable[res[0]]
            except: pass

        if found and found not in matched: matched.append(found)
        elif not found: not_found.append(raw_term)

    return matched, not_found

HEADERS = {'User-Agent': 'MediScanAI/2.0'}

@st.cache_data(ttl=300)
def geocode(location):
    try:
        r = requests.get('https://nominatim.openstreetmap.org/search',
            params={'q': f'{location}, India', 'format': 'json', 'limit': 1},
            headers=HEADERS, timeout=10).json()
        if r: return float(r[0]['lat']), float(r[0]['lon']), None
        return None, None, 'Location not found'
    except Exception as e:
        return None, None, str(e)

def haversine(lat, lon, elat, elon):
    dlat = math.radians(float(elat) - lat)
    dlon = math.radians(float(elon) - lon)
    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat)) *
         math.cos(math.radians(float(elat))) * math.sin(dlon/2)**2)
    return round(6371 * 2 * math.asin(math.sqrt(a)), 2)

@st.cache_data(ttl=300)
def find_places(location, radius_km=5, kind='hospital'):
    lat, lon, err = geocode(location)
    if err: return [], err
    rm = radius_km * 1000
    if kind == 'hospital':
        q = f'[out:json][timeout:25];(node[amenity=hospital](around:{rm},{lat},{lon});way[amenity=hospital](around:{rm},{lat},{lon});node[amenity=clinic](around:{rm},{lat},{lon}););out center 10;'
    else:
        q = f'[out:json][timeout:25];(node[amenity=doctors](around:{rm},{lat},{lon});node[amenity=clinic](around:{rm},{lat},{lon}););out 10;'
    try:
        res = requests.post('https://overpass-api.de/api/interpreter',
            data={'data': q}, headers=HEADERS, timeout=30).json()
        items = []
        for el in res.get('elements', []):
            t = el.get('tags', {})
            name = t.get('name') or t.get('name:en') or kind.title()
            elat = float(el.get('lat') or el.get('center', {}).get('lat', lat))
            elon = float(el.get('lon') or el.get('center', {}).get('lon', lon))
            items.append({
                'name': name, 'phone': t.get('phone', ''),
                'distance_km': haversine(lat, lon, elat, elon),
                'maps_url': f'https://www.google.com/maps?q={elat},{elon}'
            })
        items.sort(key=lambda x: x['distance_km'])
        return items[:5], None
    except Exception as e:
        return [], str(e)

# ══════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════

rf_metrics = M['rf']['metrics']
lstm_metrics = M['lstm']['metrics']
rf_acc = rf_metrics['top1'] * 100
lstm_acc = lstm_metrics['top1'] * 100
rf_top3 = rf_metrics['top3'] * 100
lstm_top3 = lstm_metrics['top3'] * 100
n_diseases = len(M['rf']['le'].classes_)
n_symptoms = len(M['rf']['symptoms'])
cal_gap = rf_metrics.get('cal_gap') or 0

title_col, reset_col = st.columns([5, 1])
with title_col:
    st.markdown('<h1 style="font-size:32px;font-weight:700;margin-bottom:2px">🩺 MediScan AI</h1>',
                unsafe_allow_html=True)
    st.caption(f"Intelligent Multi-Disease Detection · {n_diseases} diseases · {n_symptoms} symptoms")
with reset_col:
    st.markdown('<div style="margin-top:14px"></div>', unsafe_allow_html=True)
    if st.button('🔄 Reset', use_container_width=True):
        st.session_state.selected_region = 'all'
        st.session_state.reset_counter += 1
        st.session_state.show_results = False
        st.session_state.input_mode = 'text'
        for key in list(st.session_state.keys()):
            if key.startswith('s_') or key.startswith('cb_'):
                del st.session_state[key]
        st.rerun()

st.markdown('---')

about_html = (
    '<div class="about-card">'
    '<div class="about-title">About <span class="accent">MediScan AI</span></div>'
    '<div class="about-sub">An explainable AI system for preliminary disease screening — '
    'built on real clinical symptom data, not general language models.</div>'
    '<div class="stat-grid">'
    f'<div class="stat-box"><div class="stat-num">{n_diseases}</div><div class="stat-label">Diseases Covered</div></div>'
    f'<div class="stat-box"><div class="stat-num">{n_symptoms}</div><div class="stat-label">Symptom Features</div></div>'
    f'<div class="stat-box"><div class="stat-num">{rf_acc:.1f}%</div><div class="stat-label">RandomForest Top-1</div></div>'
    f'<div class="stat-box"><div class="stat-num">{lstm_acc:.1f}%</div><div class="stat-label">LSTM Top-1</div></div>'
    '</div>'
    '<div class="about-para">'
    f'MediScan AI is trained on <strong>real symptom-disease records</strong> covering '
    f'<strong>{n_diseases} conditions</strong> across multiple body systems — from cardiology '
    'and neurology to dermatology and psychiatry. Unlike general-purpose AI assistants, '
    'MediScan AI provides <strong>deterministic, reproducible, calibrated predictions</strong>. '
    'The same symptoms always produce the same ranked output — a fundamental requirement for '
    'any medical screening tool. All predictions happen <strong>locally on-device</strong> — '
    'your symptoms never leave your machine.'
    '</div>'
    f'<div style="font-size:12px;color:#64748b;margin-bottom:12px">'
    f'<strong>Model Performance:</strong> '
    f'RandomForest Top-3 {rf_top3:.1f}% · LSTM Top-3 {lstm_top3:.1f}% · '
    f'Calibration gap {cal_gap:.3f}'
    '</div>'
    '<div class="feature-grid">'
    '<div class="feat-box"><div class="feat-icon">🎯</div><div class="feat-title">Explainable AI</div>'
    '<div class="feat-desc">Shows which symptoms drove the prediction.</div></div>'
    '<div class="feat-box"><div class="feat-icon">📊</div><div class="feat-title">Calibrated Confidence</div>'
    '<div class="feat-desc">Scores backed by a calibration curve.</div></div>'
    '<div class="feat-box"><div class="feat-icon">💬</div><div class="feat-title">Text or Checkbox Input</div>'
    '<div class="feat-desc">Type symptoms or select from checklist.</div></div>'
    '<div class="feat-box"><div class="feat-icon">🏥</div><div class="feat-title">Hospital &amp; Doctor Finder</div>'
    '<div class="feat-desc">Nearby specialists via OpenStreetMap.</div></div>'
    '<div class="feat-box"><div class="feat-icon">🔒</div><div class="feat-title">100% Private</div>'
    '<div class="feat-desc">No health data leaves your device.</div></div>'
    '<div class="feat-box"><div class="feat-icon">⚡</div><div class="feat-title">Dual Model Comparison</div>'
    '<div class="feat-desc">RandomForest and LSTM side-by-side.</div></div>'
    '</div></div>'
)
st.markdown(about_html, unsafe_allow_html=True)

st.markdown('---')

col1, col2 = st.columns([2, 3])
with col1:
    model_choice = st.selectbox('🤖 Select Model:', ['RandomForest', 'LSTM'], key='model_sel')
with col2:
    st.markdown(
        '<div style="margin-top:8px;font-size:12px">'
        f'<span class="rf-badge">RF: {rf_acc:.1f}%</span>'
        f'<span class="lstm-badge" style="margin-left:6px">LSTM: {lstm_acc:.1f}%</span>'
        '</div>', unsafe_allow_html=True)

st.markdown('---')

st.subheader('Step 1 — Filter by Body Region')
st.caption('Select a body region to bring relevant symptoms to the top of the checklist.')

regions = ['all','head','chest','abdomen','skin','joints','general','urinary','throat']
btn_cols = st.columns(len(regions))
for i, reg in enumerate(regions):
    prefix = '✓ ' if st.session_state.selected_region == reg else ''
    if btn_cols[i].button(f'{prefix}{REGION_META[reg][0]}',
                          key=f'rb_{reg}_{st.session_state.reset_counter}',
                          use_container_width=True):
        st.session_state.selected_region = reg
        st.rerun()

active = st.session_state.selected_region
m = REGION_META[active]
st.markdown(
    f'<div style="background:{m[1]};border:1.5px solid {m[2]};color:{m[3]};'
    f'padding:6px 14px;border-radius:8px;font-size:13px;font-weight:500;'
    f'display:inline-block;margin-top:6px">Selected region: {m[0]}</div>',
    unsafe_allow_html=True)

st.markdown('---')

st.subheader('Step 2 — Enter Your Symptoms')

mode_col1, mode_col2, _ = st.columns([1, 1, 3])
with mode_col1:
    if st.button('🔤 Text Input', use_container_width=True,
                 type='primary' if st.session_state.input_mode == 'text' else 'secondary',
                 key='mode_text'):
        st.session_state.input_mode = 'text'; st.rerun()
with mode_col2:
    if st.button('☑️ Checkbox Selection', use_container_width=True,
                 type='primary' if st.session_state.input_mode == 'checkbox' else 'secondary',
                 key='mode_checkbox'):
        st.session_state.input_mode = 'checkbox'; st.rerun()

st.markdown('')

if st.session_state.input_mode == 'text':
    st.markdown(
        '<div class="section-card">'
        '<div class="section-label">🔤 Text Input</div>'
        '<div class="section-desc">Type symptoms separated by commas. '
        'Supports typos, synonyms &amp; Hindi words. E.g. <code>fever, headache, joint pain</code>'
        '</div></div>', unsafe_allow_html=True)

    text_col1, text_col2 = st.columns([4, 1])
    text_input = text_col1.text_input(
        'Symptoms:', placeholder='e.g. fever, headache, fatigue, cough',
        key=f'text_sym_{st.session_state.reset_counter}', label_visibility='collapsed')
    text_predict_clicked = text_col2.button(
        '🔍 Predict', key=f'text_predict_{st.session_state.reset_counter}',
        type='primary', use_container_width=True, disabled=not text_input.strip())

    if text_input.strip():
        matched_preview, not_found_preview = match_symptoms(text_input, M['rf']['symptoms'])
        if matched_preview:
            tags_html = ''.join([f'<span class="sym-tag">✓ {s.title()}</span>' for s in matched_preview])
            if not_found_preview:
                tags_html += ''.join([f'<span class="sym-miss">? {s}</span>' for s in not_found_preview])
            st.markdown(f'<div style="margin:6px 0">{tags_html}</div>', unsafe_allow_html=True)
            if not_found_preview:
                st.caption(f'⚠️ Not matched: {", ".join(not_found_preview)}')
        else:
            st.caption('No symptoms matched yet. Check spelling or switch to checkbox mode.')

    checkbox_selected = []
    cb_predict_btn = False
else:
    st.markdown(
        '<div class="section-card">'
        '<div class="section-label">☑️ Checkbox Selection</div>'
        '<div class="section-desc">Browse all symptoms and select multiple. '
        'Region filter above brings relevant symptoms first. Use search to narrow down.'
        '</div></div>', unsafe_allow_html=True)

    region = st.session_state.selected_region
    symptom_list_all = M['rf']['symptoms']

    if region != 'all' and region in REGION_SYMPTOMS:
        priority = [s for s in symptom_list_all if s in REGION_SYMPTOMS[region]]
        rest = [s for s in symptom_list_all if s not in REGION_SYMPTOMS[region]]
        ordered = priority + rest
        mm = REGION_META[region]
        st.markdown(
            f'<div style="background:{mm[1]};color:{mm[3]};padding:6px 14px;border-radius:8px;'
            f'font-size:12px;font-weight:500;display:inline-block;margin-bottom:8px">'
            f'Showing {len(priority)} {mm[0]} symptoms first, followed by all others</div>',
            unsafe_allow_html=True)
    else:
        ordered = symptom_list_all

    search_q = st.text_input('Search in checklist:',
                             placeholder='Type to filter... e.g. fever, pain, rash',
                             key=f'cb_search_{st.session_state.reset_counter}')
    if search_q.strip():
        q = search_q.lower()
        ordered = [s for s in ordered if q in s.lower()]
        st.caption(f'{len(ordered)} symptoms match "{search_q}"')

    if not ordered:
        st.warning('No symptoms match your search.')
        checkbox_selected = []
    else:
        c1, c2, c3 = st.columns(3)
        col_list = [c1, c2, c3]
        checkbox_selected = []
        for i, sym in enumerate(ordered):
            key = f's_{sym}_{st.session_state.reset_counter}'
            if col_list[i % 3].checkbox(sym.title(), key=key):
                checkbox_selected.append(sym)

    if checkbox_selected:
        st.success(f'**{len(checkbox_selected)} symptom(s) selected:** ' +
                   ', '.join(checkbox_selected))
        cb_predict_btn = st.button(
            f'🔍 Predict from {len(checkbox_selected)} Selected Symptoms',
            type='primary', key=f'cb_predict_{st.session_state.reset_counter}')
    else:
        st.caption('No symptoms selected yet.')
        cb_predict_btn = False

    text_input = ''
    text_predict_clicked = False

st.markdown('---')
st.subheader('Step 3 — Location (Optional)')
st.caption('For finding nearby hospitals and specialists.')

loc1, loc2 = st.columns(2)
area = loc1.text_input('Area / Locality:', placeholder='e.g. Vijay Nagar',
                       key=f'area_{st.session_state.reset_counter}')
city = loc2.text_input('City:', placeholder='e.g. Bhopal',
                       key=f'city_{st.session_state.reset_counter}')
full_loc = ', '.join(filter(None, [area.strip(), city.strip()]))

if full_loc:
    st.caption(f'Will search near: **{full_loc}**')

final_symptoms = []
predict_source = None

if st.session_state.input_mode == 'text' and text_predict_clicked and text_input.strip():
    matched, _ = match_symptoms(text_input, M['rf']['symptoms'])
    final_symptoms = matched
    predict_source = 'text'
    st.session_state.show_results = True
elif st.session_state.input_mode == 'checkbox' and cb_predict_btn and checkbox_selected:
    final_symptoms = checkbox_selected
    predict_source = 'checkbox'
    st.session_state.show_results = True

if st.session_state.show_results and final_symptoms:
    st.markdown('---')
    st.subheader('🎯 Prediction Results')

    source_label = '🔤 Text Input' if predict_source == 'text' else '☑️ Checkbox Selection'
    st.markdown(
        f'<div style="background:#EAF3DE;color:#27500A;padding:6px 14px;'
        f'border-radius:8px;font-size:12px;font-weight:500;display:inline-block;margin-bottom:12px">'
        f'Source: {source_label} · {len(final_symptoms)} symptom(s) used</div>',
        unsafe_allow_html=True)

    if model_choice == 'RandomForest' or M['lstm']['model'] is None:
        vec = np.zeros(len(M['rf']['symptoms']))
        for s in final_symptoms:
            if s in M['rf']['symptoms']:
                vec[M['rf']['symptoms'].index(s)] = 1
        proba = M['rf']['model'].predict_proba(pd.DataFrame([vec], columns=M['rf']['symptoms']))[0]
        le_use = M['rf']['le']
        used_model = 'RandomForest'
    else:
        seq = [M['lstm']['sym_to_idx'][s] for s in final_symptoms if s in M['lstm']['sym_to_idx']]
        seq = seq[:M['lstm']['max_len']]
        seq = seq + [0] * (M['lstm']['max_len'] - len(seq))
        proba = M['lstm']['model'].predict(np.array([seq]), verbose=0)[0]
        le_use = M['lstm']['le']
        used_model = 'LSTM'

    top3_idx = np.argsort(proba)[::-1][:3]
    medals = ['🥇', '🥈', '🥉']

    pred_cols = st.columns(3)
    for rank, idx in enumerate(top3_idx):
        disease = le_use.classes_[idx]
        conf = proba[idx] * 100
        spec = get_specialist(disease)

        if conf >= 60:   bg, urgency = '#FCEBEB', '🔴 High'
        elif conf >= 30: bg, urgency = '#FAEEDA', '🟡 Medium'
        else:            bg, urgency = '#E6F1FB', '🔵 Low'

        with pred_cols[rank]:
            card_html = (
                f'<div class="pred-card" style="background:{bg}">'
                f'<div style="font-size:22px">{medals[rank]}</div>'
                f'<div style="font-size:15px;font-weight:700;margin:4px 0">{disease}</div>'
                f'<div style="font-size:26px;font-weight:700;color:#534AB7">{conf:.1f}%</div>'
                f'<div class="conf-bar"><div class="conf-fill" style="width:{min(conf,100)}%"></div></div>'
                f'<div style="font-size:12px;color:#555;margin-top:8px">👨‍⚕️ {spec}</div>'
                f'<div style="font-size:11px;color:#666;margin-top:4px">{urgency}</div>'
                f'</div>')
            st.markdown(card_html, unsafe_allow_html=True)

    # Key Symptoms Driving Prediction
    if model_choice == 'RandomForest' or M['lstm']['model'] is None:
        feat_imp_all = M['rf']['model'].feature_importances_
        top_syms = sorted(
            [(s, float(feat_imp_all[M['rf']['symptoms'].index(s)]))
             for s in final_symptoms if s in M['rf']['symptoms']],
            key=lambda x: x[1], reverse=True
        )[:5]
    else:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            rf_proxy = joblib.load(os.path.join(base_dir, 'models', 'rf_model_compressed.joblib'))
            rf_syms_proxy = pickle.load(open(os.path.join(base_dir, 'models', 'rf_symptom_list.pkl'), 'rb'))
            feat_imp_all = rf_proxy.feature_importances_
            top_syms = sorted(
                [(s, float(feat_imp_all[rf_syms_proxy.index(s)]))
                 for s in final_symptoms if s in rf_syms_proxy],
                key=lambda x: x[1], reverse=True
            )[:5]
        except:
            top_syms = []

    if top_syms:
        st.markdown('#### 🔬 Key Symptoms Driving This Prediction')
        st.caption('Showing contribution of your entered symptoms based on model feature importance.')

        total_imp = sum(v for _, v in top_syms) or 1
        sym_cols = st.columns(min(len(top_syms), 5))

        for i, (sym, imp) in enumerate(top_syms):
            pct = round(imp / total_imp * 100, 1)
            with sym_cols[i]:
                card_html = (
                    f'<div class="key-sym-card">'
                    f'<div style="font-size:12px;font-weight:500;color:#1e293b">'
                    f'{sym.title()}</div>'
                    f'<div style="font-size:20px;font-weight:700;color:#534AB7;margin:4px 0">'
                    f'{pct}%</div>'
                    f'<div style="background:#EEEDFE;border-radius:4px;height:6px">'
                    f'<div style="background:#534AB7;width:{pct}%;height:6px;border-radius:4px">'
                    f'</div></div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

    st.session_state.history.append({
        'time': pd.Timestamp.now().strftime('%H:%M:%S'),
        'symptoms': ', '.join(final_symptoms[:5]) + ('...' if len(final_symptoms) > 5 else ''),
        'prediction': le_use.classes_[top3_idx[0]],
        'confidence': f"{proba[top3_idx[0]]*100:.1f}%",
        'model': used_model
    })

    if full_loc:
        st.markdown('---')
        st.subheader('🏥 Nearby Hospitals & Doctors')
        st.caption(f'Searching near **{full_loc}** · OpenStreetMap')

        top_spec = get_specialist(le_use.classes_[top3_idx[0]])
        doc_col, hosp_col = st.columns(2)

        with doc_col:
            st.markdown(f'##### 👨‍⚕️ {top_spec}s Near You')
            with st.spinner('Searching doctors...'):
                docs, err = find_places(full_loc, 8, 'doctors')
            if err: st.warning(f'Search failed: {err}')
            elif docs:
                for d in docs:
                    st.markdown(
                        f'<div class="place-card">'
                        f'<div style="font-weight:600">{d["name"]}</div>'
                        f'<div style="font-size:12px;color:#666">📍 {d["distance_km"]} km away</div>'
                        f'<a href="{d["maps_url"]}" target="_blank" style="font-size:12px;color:#534AB7">Open in Maps →</a>'
                        f'</div>', unsafe_allow_html=True)
            else: st.info('No doctors found.')

        with hosp_col:
            st.markdown('##### 🏥 Hospitals Near You')
            with st.spinner('Searching hospitals...'):
                hosps, err = find_places(full_loc, 5, 'hospital')
            if err: st.warning(f'Search failed: {err}')
            elif hosps:
                for h in hosps:
                    st.markdown(
                        f'<div class="place-card">'
                        f'<div style="font-weight:600">{h["name"]}</div>'
                        f'<div style="font-size:12px;color:#666">📍 {h["distance_km"]} km away</div>'
                        f'<a href="{h["maps_url"]}" target="_blank" style="font-size:12px;color:#534AB7">Open in Maps →</a>'
                        f'</div>', unsafe_allow_html=True)
            else: st.info('No hospitals found.')

    st.markdown('---')
    st.warning('⚕️ **Medical Disclaimer:** MediScan AI is for educational and preliminary '
               'screening purposes only. NOT a substitute for professional medical diagnosis. '
               'Always consult a qualified healthcare provider. Emergency: call 108.')

if st.session_state.history:
    st.markdown('---')
    with st.expander(f'📜 Prediction History ({len(st.session_state.history)} entries)'):
        for h in reversed(st.session_state.history[-10:]):
            st.markdown(f"**{h['time']}** · {h['prediction']} ({h['confidence']}) · _{h['model']}_")
            st.caption(f"Symptoms: {h['symptoms']}")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ============================================================
# FINAL VISIBILITY STYLING FOR ALL BLOCKS 
# ============================================================
st.markdown("""
<style>
    /* 1. Force the dark background gradient for the main app */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    
    /* 2. Force the sidebar background to match your dark theme */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    
    /* 3. UNIVERSAL WHITE TEXT: Force headers, text fields, sidebar items, and bullets to white */
    .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li,
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] li {
        color: #ffffff !important;
    }
    
    /* 4. DYNAMIC CARDS FIX: Force all text inside your results cards/containers to be clean white */
    div[data-testid="stMarkdownContainer"] * {
        color: #ffffff !important;
    }
    
    /* 5. INPUT FIELDS CONTRAST: Keep choices INSIDE dropdowns, selection fields, and inputs dark gray */
    div[data-baseweb="select"] *, 
    div[role="listbox"] *, 
    .stSelectbox div, 
    input {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)
# ============================================================
# LOAD MODEL AND PREPROCESSING ARTIFACTS
# ============================================================

@st.cache_resource
def load_models():
    """Load model and scalers (cached for performance)"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        model = joblib.load(os.path.join(base_dir, 'stroke_model.pkl'))
        std_scaler = joblib.load(os.path.join(base_dir, 'std_scaler.pkl'))
        minmax_scaler = joblib.load(os.path.join(base_dir, 'minmax_scaler.pkl'))
        feature_names = joblib.load(os.path.join(base_dir, 'features.pkl'))
        return model, std_scaler, minmax_scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"""
        **Model files not found!** 🚨

        You need to upload the following files to the same directory:
        - `stroke_model.pkl`
        - `std_scaler.pkl`
        - `minmax_scaler.pkl`
        - `features.pkl`

        Error: {str(e)}

        ### How to get these files:
        1. Run your training notebook in Google Colab
        2. After training, the .pkl files will be created
        3. Download them and upload to your GitHub repository
        """)
        st.stop()

# Load the models
model, std_scaler, minmax_scaler, feature_names = load_models()

# Dataset statistics (from training data after preprocessing)
BMI_MIN = 10.3
BMI_MAX = 47.5
GLUCOSE_MIN = 55.12
GLUCOSE_MAX = 169.46

# ============================================================
# CUSTOM STYLES
# ============================================================

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0c1929 0%, #1a365d 50%, #0f172a 100%);
    }

    /* Title styling */
    .main-title {
        font-size: 3rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #fff 0%, #0ea5e9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 1rem 0;
    }

    /* Card styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    /* Risk level badges */
    .risk-low {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
    }

    .risk-medium {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
    }

    .risk-high {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
    }

    /* Input fields */
    .stNumberInput input, .stSelectbox select {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #0ea5e9, #06b6d4) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4) !important;
    }

    /* Success message */
    .element-container .stSuccess {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
    }

    /* Info boxes */
    .info-box {
        background: rgba(14, 165, 233, 0.1);
        border-left: 4px solid #0ea5e9;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Progress bar styling */
    .stProgress > div > div {
        background: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def preprocess_input(data):
    """
    Preprocess input data to match training pipeline
    """
    df = pd.DataFrame([data])

    # StandardScaler for age
    df['age'] = std_scaler.transform(df[['age']])

    # MinMaxScaler for avg_glucose_level
    df['avg_glucose_level'] = minmax_scaler.transform(df[['avg_glucose_level']])

    # Manual MinMax for BMI
    df['bmi'] = (df['bmi'] - BMI_MIN) / (BMI_MAX - BMI_MIN)
    df['bmi'] = df['bmi'].clip(0, 1)

    # Label encode ever_married (Yes=1, No=0)
    df['ever_married'] = 1 if df['ever_married'].iloc[0] == 'Yes' else 0

    # Label encode Residence_type (Urban=1, Rural=0)
    df['Residence_type'] = 1 if df['Residence_type'].iloc[0] == 'Urban' else 0

    # One-hot encoding for gender
    gender = df['gender'].iloc[0]
    df['gender_Female'] = 1 if gender == 'Female' else 0
    df['gender_Male'] = 1 if gender == 'Male' else 0
    df['gender_Other'] = 1 if gender == 'Other' else 0

    # One-hot encoding for work type
    work_type = df['work_type'].iloc[0]
    df['work_type_Govt_job'] = 1 if work_type == 'Govt_job' else 0
    df['work_type_Never_worked'] = 1 if work_type == 'Never_worked' else 0
    df['work_type_Private'] = 1 if work_type == 'Private' else 0
    df['work_type_Self-employed'] = 1 if work_type == 'Self-employed' else 0
    df['work_type_children'] = 1 if work_type == 'children' else 0

    # One-hot encoding for smoking status
    smoking = df['smoking_status'].iloc[0]
    df['smoking_status_Unknown'] = 1 if smoking == 'Unknown' else 0
    df['smoking_status_formerly smoked'] = 1 if smoking == 'formerly smoked' else 0
    df['smoking_status_never smoked'] = 1 if smoking == 'never smoked' else 0
    df['smoking_status_smokes'] = 1 if smoking == 'smokes' else 0

    # Select features in correct order
    features_df = df[feature_names]

    return features_df


def get_risk_level(probability):
    """Classify risk level"""
    if probability < 0.20:
        return 'LOW', '#10b981', '✅'
    elif probability < 0.50:
        return 'MEDIUM', '#f59e0b', '⚠️'
    else:
        return 'HIGH', '#ef4444', '🚨'


def get_risk_factors(data):
    """Identify risk factors"""
    factors = []
    if data['age'] > 55:
        factors.append(('👴 Advanced Age (>55 years)', 'high'))
    if data['hypertension'] == 1:
        factors.append(('🩺 Hypertension', 'high'))
    if data['heart_disease'] == 1:
        factors.append(('❤️ Heart Disease', 'high'))
    if data['avg_glucose_level'] > 126:
        factors.append(('🩸 High Glucose Level (>126 mg/dL)', 'medium'))
    if data['bmi'] > 30:
        factors.append(('⚖️ Obesity (BMI > 30)', 'medium'))
    if data['smoking_status'] in ['smokes', 'formerly smoked']:
        factors.append(('🚬 Smoking History', 'medium'))
    return factors if factors else [('✅ No Major Risk Factors Identified', 'low')]


def get_recommendations(risk_level):
    """Get personalized recommendations"""
    if risk_level == 'HIGH':
        return [
            '🏥 Consult a healthcare professional immediately',
            '📊 Monitor blood pressure daily',
            '🥗 Follow a heart-healthy diet (low sodium, high fiber)',
            '🏃 Begin supervised exercise program',
            '💊 Review medications with your doctor'
        ]
    elif risk_level == 'MEDIUM':
        return [
            '👨‍⚕️ Schedule a check-up with your doctor',
            '📈 Monitor blood pressure regularly',
            '🥬 Improve diet: reduce salt and saturated fats',
            '🚴 Start moderate exercise (30 min/day)',
            '🚭 Quit smoking if applicable'
        ]
    else:
        return [
            '✅ Maintain current healthy lifestyle',
            '📅 Continue regular health check-ups',
            '💪 Stay physically active',
            '🥑 Maintain a balanced diet',
            '🔍 Monitor health metrics annually'
        ]

# ============================================================
# MAIN APPLICATION
# ============================================================

# Title
st.markdown('<h1 class="main-title">AI-Powered Stroke Risk Assessment</h1>', unsafe_allow_html=True)

# Subtitle
st.markdown("""
<p style="text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;">
Advanced machine learning model to predict stroke risk. Get personalized insights instantly.
</p>
""", unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([2, 1])

# ============================================================
# INPUT FORM (Left Column)
# ============================================================

with col1:
    st.markdown("### 📋 Patient Information")

    # Create input columns
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input(
            "🎂 Age (years)",
            value=45,
            help="Patient's age in years"
        )

        gender = st.selectbox(
            "👤 Gender",
            options=['Male', 'Female', 'Other'],
            help="Patient's gender"
        )

        hypertension = st.checkbox(
            "🩺 Hypertension",
            help="Does the patient have hypertension?"
        )

        heart_disease = st.checkbox(
            "❤️ Heart Disease",
            help="Does the patient have heart disease?"
        )

    with c2:
        ever_married = st.selectbox(
            "💍 Ever Married",
            options=['Yes', 'No'],
            help="Has the patient ever been married?"
        )

        work_type = st.selectbox(
            "💼 Work Type",
            options=['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
            help="Patient's work type"
        )

        residence_type = st.selectbox(
            "🏠 Residence Type",
            options=['Urban', 'Rural'],
            help="Type of residence area"
        )

        avg_glucose_level = st.number_input(
            "🩸 Avg Glucose Level (mg/dL)",
            value=85.0,
            step=0.1,
            help="Average glucose level in blood"
        )

        bmi = st.number_input(
            "⚖️ BMI",
            value=25.0,
            step=0.1,
            help="Body Mass Index"
        )

    smoking_status = st.selectbox(
        "🚬 Smoking Status",
        options=['never smoked', 'formerly smoked', 'smokes', 'Unknown'],
        help="Patient's smoking history"
    )

    # Predict button
    predict_btn = st.button("🔮 Analyze Stroke Risk", type="primary", use_container_width=True)

# ============================================================
# SIDEBAR (Model Info)
# ============================================================

with st.sidebar:
    st.markdown("## 🤖 Model Information")

    st.markdown("""
    <div class="metric-card">
        <p style="color: #0ea5e9; font-size: 1.5rem; font-weight: bold; margin: 0;">Logistic Regression</p>
        <p style="color: #94a3b8; font-size: 0.9rem;">Algorithm Used</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Accuracy", "79.2%")
        st.metric("Recall", "48.5%")

    with col_b:
        st.metric("F1-Score", "14.9%")
        st.metric("Training Size", "5,110")

    st.markdown("---")

    st.markdown("### 💡 Prevention Tips")

    tips = [
        "🍎 Eat fruits & vegetables daily",
        "🏃 Exercise 30 min/day",
        "🚭 Avoid smoking",
        "🩺 Monitor blood pressure",
        "😴 Get adequate sleep"
    ]

    for tip in tips:
        st.markdown(f"- {tip}")

    st.markdown("---")

    st.markdown("""
    <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <p style="color: #ef4444; font-weight: 600; margin-bottom: 0.5rem;">⚠️ Disclaimer</p>
        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
        This tool is for educational purposes only. Not a substitute for professional medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PREDICTION RESULTS (Right Column or Below)
# ============================================================

with col2:
    st.markdown("### 📊 Results")

    if predict_btn:
        # Prepare data
        input_data = {
            'age': age,
            'gender': gender,
            'hypertension': 1 if hypertension else 0,
            'heart_disease': 1 if heart_disease else 0,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        try:
            # Preprocess and predict
            processed_data = preprocess_input(input_data)
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]

            # Get risk level
            risk_level, risk_color, risk_icon = get_risk_level(probability)

            # Display probability
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <p style="font-size: 3rem; font-weight: 700; color: {}; margin: 0;">{:.1f}%</p>
                <p style="color: #94a3b8; font-size: 0.9rem;">Stroke Probability</p>
            </div>
            """.format(risk_color, probability * 100), unsafe_allow_html=True)

            # Risk level badge
            risk_class = f'risk-{risk_level.lower()}'
            st.markdown(f"""
            <div class="{risk_class}" style="margin: 1rem 0;">
                {risk_icon} {risk_level} RISK
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.progress(probability)

            # Risk factors
            st.markdown("#### Identified Risk Factors")
            factors = get_risk_factors(input_data)

            for factor, severity in factors:
                severity_color = '#ef4444' if severity == 'high' else ('#f59e0b' if severity == 'medium' else '#10b981')
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.5); border-left: 4px solid {severity_color};
                     padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    {factor}
                </div>
                """, unsafe_allow_html=True)

            # Recommendations
            st.markdown("#### 📝 Recommendations")
            recommendations = get_recommendations(risk_level)

            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div style="background: rgba(14, 165, 233, 0.1); padding: 0.75rem;
                     border-radius: 8px; margin-bottom: 0.5rem;">
                    <span style="background: linear-gradient(135deg, #0ea5e9, #06b6d4);
                          color: white; padding: 0.2rem 0.6rem; border-radius: 50%;
                          font-size: 0.8rem; font-weight: 600; margin-right: 0.5rem;">{i}</span>
                    {rec}
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.code(str(e))

    else:
        st.info("👈 Fill in the patient information and click 'Analyze Stroke Risk' to see results.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #64748b; font-size: 0.85rem;">
    Built with ❤️ using Streamlit | Logistic Regression Model |
    <a href="https://github.com" style="color: #0ea5e9;">View on GitHub</a>
</p>
""", unsafe_allow_html=True)

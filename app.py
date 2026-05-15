import streamlit as st
import joblib
import pandas as pd
import os
import json

# --- Page Config ---
st.set_page_config(
    page_title="Rice Blast Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4CAF50, #00BCD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        padding-top: 20px;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: #888888;
        margin-bottom: 40px;
    }
    
    /* Card styling for results */
    .result-card-healthy {
        background: linear-gradient(135deg, #1B5E20 0%, #4CAF50 100%);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        text-align: center;
        animation: fadeIn 0.8s ease-out;
        border: 1px solid #81C784;
    }
    
    .result-card-disease {
        background: linear-gradient(135deg, #b71c1c 0%, #e53935 100%);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        text-align: center;
        animation: fadeIn 0.8s ease-out;
        border: 1px solid #ef5350;
    }
    
    .result-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .result-prob {
        font-size: 1.8rem;
        font-weight: 600;
        color: #F5F5F5;
        background: rgba(0,0,0,0.2);
        padding: 10px 20px;
        border-radius: 10px;
        display: inline-block;
    }
    
    .result-desc {
        font-size: 1.1rem;
        margin-top: 25px;
        font-weight: 400;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    model_path = os.path.join(os.path.dirname(__file__), 'mlp_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        return None, None

model, scaler = load_assets()

# --- Main UI ---
st.markdown('<div class="main-header">Rice Blast Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced ML-based crop health monitoring system using environmental and physical indicators.</div>', unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("⚠️ Model or Scaler not found! Please run `train_model.py` first to generate `mlp_model.pkl` and `scaler.pkl`.")
    st.stop()

# --- Sidebar Inputs ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3014/3014411.png", width=120)
    st.title("Field Observations")
    st.markdown("Adjust the environmental conditions to predict the likelihood of Rice Blast disease.")
    
    st.markdown("### Environmental Factors")
    temperature = st.slider("Temperature (°C)", min_value=10.0, max_value=45.0, value=28.0, step=0.1, help="Current ambient temperature.")
    humidity = st.slider("Humidity (%)", min_value=30.0, max_value=100.0, value=85.0, step=1.0, help="Relative humidity percentage.")
    leaf_wetness = st.slider("Leaf Wetness (Hours)", min_value=0.0, max_value=24.0, value=12.0, step=0.5, help="Duration of leaf wetness.")
    
    st.markdown("### Crop Conditions")
    lesion_size = st.slider("Lesion Size (cm)", min_value=0.0, max_value=15.0, value=3.0, step=0.1, help="Average size of lesions on the leaves.")
    nitrogen_level = st.slider("Nitrogen Level (kg/ha)", min_value=10.0, max_value=200.0, value=60.0, step=1.0, help="Amount of nitrogen fertilizer applied.")
    
    st.markdown("---")
    predict_btn = st.button("Analyze Health Status", use_container_width=True, type="primary")

# --- Create Tabs ---
tab1, tab2 = st.tabs(["Health Prediction", "Model Performance Metrics"])

with tab1:
    # --- Layout for Data display ---
    st.markdown("### Current Sensor Readings")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Temperature", f"{temperature:.1f} °C")
    col2.metric("Humidity", f"{humidity:.1f} %")
    col3.metric("Leaf Wetness", f"{leaf_wetness:.1f} hrs")
    col4.metric("Lesion Size", f"{lesion_size:.1f} cm")
    col5.metric("Nitrogen Level", f"{nitrogen_level:.1f} kg/ha")
    
    st.markdown("---")
    
    # --- Prediction Logic ---
    if predict_btn:
        with st.spinner("Analyzing parameters using MLP Neural Network..."):
            # Prepare input data
            feature_names = ['Temperature', 'Humidity', 'Leaf_Wetness', 'Lesion_Size', 'Nitrogen_Level']
            input_data = pd.DataFrame([[temperature, humidity, leaf_wetness, lesion_size, nitrogen_level]], columns=feature_names)
            
            # Scale data
            scaled_data = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0]
            confidence = probability[prediction] * 100
            
            # Display Result
            st.markdown("### Prediction Analysis")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card-disease">
                    <div class="result-title">⚠️ Rice Blast Detected</div>
                    <div class="result-prob">Confidence: {confidence:.2f}%</div>
                    <div class="result-desc" style="color: #FFCDD2;">
                        Conditions are highly favorable for Rice Blast. <br/>
                        Immediate action and targeted fungicide application are recommended to prevent yield loss.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div class="result-card-healthy">
                    <div class="result-title">✅ Crop is Healthy</div>
                    <div class="result-prob">Confidence: {confidence:.2f}%</div>
                    <div class="result-desc" style="color: #C8E6C9;">
                        No critical signs of Rice Blast disease detected. <br/>
                        Current environmental parameters are within safe thresholds for the crop.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()

with tab2:
    st.markdown("### Model Performance Overview")
    metrics_path = os.path.join(os.path.dirname(__file__), 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        m3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        m4.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
        
        st.markdown("---")
        st.markdown("###Visualizations")
        
        col1, col2 = st.columns(2)
        plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
        
        with col1:
            if os.path.exists(os.path.join(plots_dir, 'confusion_matrix.png')):
                st.image(os.path.join(plots_dir, 'confusion_matrix.png'), caption="Confusion Matrix")
            if os.path.exists(os.path.join(plots_dir, 'class_distribution.png')):
                st.image(os.path.join(plots_dir, 'class_distribution.png'), caption="Class Distribution")
                
        with col2:
            if os.path.exists(os.path.join(plots_dir, 'correlation_matrix.png')):
                st.image(os.path.join(plots_dir, 'correlation_matrix.png'), caption="Feature Correlation")
    else:
        st.warning("⚠️ Metrics not found! Please run `train_model.py` to generate the evaluation metrics.")

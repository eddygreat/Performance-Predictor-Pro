import streamlit as st
import pandas as pd
import os
import model_engine as engine
import ui_components as ui

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="wide")

# --- INITIALIZE ---
ui.apply_custom_css()

try:
    df, model, X_train_cols, feature_stats, raw_features, explainer = engine.load_model_assets()
except Exception as e:
    st.error(f"⚠️ Critical Error loading assets: {e}")
    st.stop()

# --- HEADER ---
ui.render_header()

# --- SIDEBAR ---
st.sidebar.header("🧭 Navigation")
mode = st.sidebar.radio("Choose Mode", ["Single Prediction", "Select Existing Student", "Batch Prediction (CSV)"])

# --- MODE 1 & 2: SINGLE PREDICTIONS ---
if mode in ["Single Prediction", "Select Existing Student"]:
    if mode == "Select Existing Student":
        student_idx = st.sidebar.number_input("Enter Student ID (0-648)", min_value=0, max_value=len(df)-1, value=0)
        student_data = df.iloc[student_idx].to_dict()
        st.info(f"📍 Loaded Student ID {student_idx}. Actual Result (G3): {student_data['G3']}")
    else:
        # Default starting values
        student_data = feature_stats.copy()

    # --- INPUT FORM ---
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🏠 Demographics")
            school = st.selectbox("School", ["GP", "MS"], index=0 if student_data.get('school') == 'GP' else 1)
            sex = st.selectbox("Sex", ["F", "M"], index=0 if student_data.get('sex') == 'F' else 1)
            age = st.slider("Age", 15, 22, int(student_data.get('age', 16)))
            address = st.selectbox("Address", ["U (Urban)", "R (Rural)"], index=0 if student_data.get('address') == 'U' else 1)
            famsize = st.selectbox("Family Size", ["GT3", "LE3"], index=0 if student_data.get('famsize') == 'GT3' else 1)
        with col2:
            st.subheader("📚 Academic")
            medu = st.selectbox("Mother's Education (0-4)", [0, 1, 2, 3, 4], index=int(student_data.get('Medu', 2)))
            fedu = st.selectbox("Father's Education (0-4)", [0, 1, 2, 3, 4], index=int(student_data.get('Fedu', 2)))
            studytime = st.slider("Study Time (1-4)", 1, 4, int(student_data.get('studytime', 2)))
            failures = st.slider("Past Failures (0-3)", 0, 3, int(student_data.get('failures', 0)))
        with col3:
            st.subheader("🍷 Lifestyle")
            dalc = st.slider("Workday Alcohol (1-5)", 1, 5, int(student_data.get('Dalc', 1)))
            walc = st.slider("Weekend Alcohol (1-5)", 1, 5, int(student_data.get('Walc', 1)))
            health = st.slider("Health Status (1-5)", 1, 5, int(student_data.get('health', 3)))
            absences = st.number_input("Absences", 0, 93, int(student_data.get('absences', 0)))

    if st.button("🚀 Predict Performance"):
        input_dict = student_data.copy()
        # Update with form values
        input_dict.update({
            'school': school, 'sex': sex, 'age': age, 'address': address[0],
            'famsize': famsize, 'Medu': medu, 'Fedu': fedu, 'studytime': studytime,
            'failures': failures, 'Dalc': dalc, 'Walc': walc, 'health': health, 'absences': absences
        })
        
        input_df = pd.DataFrame([input_dict])
        X_final = engine.get_prediction_data(input_df, raw_features, feature_stats, X_train_cols)
        
        prediction = model.predict(X_final)[0]
        prob = model.predict_proba(X_final)[0]
        
        st.divider()
        res1, res2 = st.columns([1, 2])
        with res1:
            if prediction == 0:
                st.success("### Prediction: PASS 🎉")
            else:
                st.error("### Prediction: FAIL ⚠️")
            st.metric("Confidence", f"{max(prob)*100:.1f}%")
        
        with res2:
            # SHAP Explanation
            shap_values = engine.get_shap_values(explainer, X_final)
            ui.render_shap_plot(shap_values, X_final)

# --- MODE 3: BATCH PREDICTION ---
elif mode == "Batch Prediction (CSV)":
    st.header("📂 Batch Processing")
    uploaded_file = st.file_uploader("Upload Student Data CSV", type="csv")
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} records.")
        
        if st.button("⚡ Run Large-Scale Prediction"):
            X_final_batch = engine.get_prediction_data(batch_df, raw_features, feature_stats, X_train_cols)
            preds = model.predict(X_final_batch)
            probs = model.predict_proba(X_final_batch)
            
            # Enrich original DF
            batch_df['Prediction'] = ["PASS" if p == 0 else "FAIL" for p in preds]
            batch_df['Confidence'] = [f"{max(pr)*100:.1f}%" for pr in probs]
            
            ui.display_metrics(batch_df)
            ui.render_batch_charts(batch_df)
            
            with st.expander("📄 View Full Results Table"):
                st.dataframe(batch_df)
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Enriched CSV", csv, "predictions.csv", "text/csv")

# --- DATA SAMPLER ---
st.divider()
with st.expander("📊 Dataset Reference Sample"):
    st.write(df.head(10))
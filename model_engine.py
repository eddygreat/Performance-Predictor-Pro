import pandas as pd
import joblib
import os
import numpy as np
import difflib
import shap
import streamlit as st

# Paths (Relative for cross-platform deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student-per.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_xgboost_model_v2.joblib")
COLS_PATH = os.path.join(BASE_DIR, "feature_columns.joblib")
STATS_PATH = os.path.join(BASE_DIR, "feature_stats.joblib")

FEATURE_ALIASES = {
    'sex': ['gender', 'sex', 'is_male', 'male', 'female'],
    'age': ['how_old', 'years', 'age'],
    'address': ['residence', 'address', 'living_area', 'location'],
    'famsize': ['family_size', 'members', 'famsize'],
    'Pstatus': ['parents_status', 'pstatus', 'cohabitation'],
    'Medu': ['mother_education', 'medu', 'm_edu'],
    'Fedu': ['father_education', 'fedu', 'f_edu'],
    'studytime': ['study_hours', 'weekly_study', 'studytime'],
    'failures': ['past_failures', 'class_failures', 'failures'],
    'absences': ['missed_classes', 'absences', 'days_off'],
    'health': ['health_status', 'health_condition', 'health'],
    'Dalc': ['workday_alcohol', 'dalc', 'weekday_drinking'],
    'Walc': ['weekend_alcohol', 'walc', 'weekend_drinking'],
    'freetime': ['free_time', 'leisure'],
    'goout': ['going_out', 'socializing']
}

@st.cache_resource
def load_model_assets():
    """Loads all ML assets and returns them."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    X_train_cols = joblib.load(COLS_PATH)
    feature_stats = joblib.load(STATS_PATH)
    raw_features = list(feature_stats.keys())
    
    # Initialize SHAP explainer
    explainer = shap.Explainer(model)
    
    return df, model, X_train_cols, feature_stats, raw_features, explainer

def map_robust_columns(input_df, target_features, feature_stats):
    """Maps arbitrary input columns to expected model features."""
    mapped_df = pd.DataFrame(index=input_df.index)
    for feature in target_features:
        found = False
        aliases = FEATURE_ALIASES.get(feature, [feature])
        for alias in aliases:
            matching_cols = [c for c in input_df.columns if c.lower() == alias.lower()]
            if matching_cols:
                mapped_df[feature] = input_df[matching_cols[0]]
                found = True
                break
        if not found:
            close_matches = difflib.get_close_matches(feature, input_df.columns, n=1, cutoff=0.6)
            if close_matches:
                mapped_df[feature] = input_df[close_matches[0]]
                found = True
        if not found:
            mapped_df[feature] = feature_stats.get(feature, 0)
    return mapped_df

def preprocess_features(df):
    """Adds interaction terms and engineered features."""
    def get_val(col, default=0):
        return df[col] if col in df.columns else default
    df['failure_risk'] = get_val('failures') * (get_val('absences') + 1)
    df['study_efficiency'] = get_val('studytime') / (get_val('absences') + 1)
    df['alcohol_burden'] = get_val('Dalc') + get_val('Walc')
    df['log_absences'] = np.log1p(get_val('absences', 0).astype(float))
    return df

def get_prediction_data(input_df, raw_features, feature_stats, X_train_cols):
    """Processes raw input into model-ready features."""
    mapped = map_robust_columns(input_df, raw_features, feature_stats)
    processed = preprocess_features(mapped)
    dummies = pd.get_dummies(processed)
    
    X_final = pd.DataFrame(columns=X_train_cols)
    X_final = pd.concat([X_final, dummies], axis=0).fillna(0)
    X_final = X_final[X_train_cols]
    return X_final

def get_shap_values(explainer, X_input):
    """Calculates SHAP values for an input instance."""
    shap_values = explainer(X_input)
    return shap_values

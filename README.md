# 🎓 Student Success Predictor

A professional-grade, AI-powered dashboard designed to identify students at risk of academic failure using advanced machine learning and explainable AI techniques.

### 🌐 Live Prototype
The application is deployed and can be accessed in real-time here:  
**🚀 [performance-predictor-pro.streamlit.app](https://performance-predictor-pro.streamlit.app/)**

## 🚀 Project Overview
The **Student Success Predictor** leverages an optimized XGBoost model to analyze demographic, social, and academic data. Its primary goal is to provide educators with a proactive tool for identifying students who may need additional support, moving from reactive grades to proactive intervention.

### ✨ Key Features
- **Proprietary XGBoost Model (v2)**: Optimized with `scale_pos_weight` to correctly identify the minority "FAIL" class ( Recall-focused).
- **🔍 Explainable AI (SHAP)**: Every single prediction is accompanied by a SHAP breakdown chart, showing exactly *why* the model predicted a certain outcome.
- **🛡️ Robust Data Ingestion**: A "Structure Agnostic" CSV upload system that uses synonym mapping and fuzzy logic to handle non-standard column names (e.g., "Gender" vs "sex").
- **📊 Advanced Analytics**: Interactive Plotly dashboards for batch predictions, featuring distribution pie charts, confidence histograms, and correlation box plots.
- **💎 Premium UI**: A modern, responsive Streamlit interface with high-end CSS styling and professional metric cards.

## 🛠️ Technical Architecture
The project follows a **Three-Tier Modular Architecture** for maintainability and scalability:

- **`app.py`**: The orchestration layer (View). Lightweight script handling the Streamlit session and user interactions.
- **`model_engine.py`**: The logic layer (Controller). Handles model loading, robust preprocessing, engineered features, and SHAP calculations.
- **`ui_components.py`**: The presentation layer (View Components). Centralizes all CSS, custom HTML, and Plotly visualization logic.

## 📂 Project Structure
```text
Performance_Predictor/
├── app.py                     # Main Entry Point
├── model_engine.py            # AI & Preprocessing Logic
├── ui_components.py           # UI & Visualization Logic
├── best_xgboost_model_v2.joblib # Optimized ML Model (Production)
├── feature_columns.joblib     # Feature Schema
├── feature_stats.joblib       # Imputation Statistics (Median/Mode)
└── README.md                  # Project Documentation
```

## 📥 Getting Started

### Prerequisites
- Python 3.8+
- [The Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance) (`student-per.csv` expected in parent directory).

### Installation
1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install streamlit pandas xgboost joblib shap plotly streamlit-shap numpy
   ```

### Running the App
From the `Performance_Predictor` directory, run:
```bash
streamlit run app.py
```

## 🧠 Data & Model Details
- **Dataset**: UCI Student Performance Data (Portuguese course).
- **Target**: Binary Classification (PASS: G3 >= 10, FAIL: G3 < 10).
- **Feature Engineering**: Includes interaction terms like `failure_risk` (`failures * absences`) and `study_efficiency`.
- **Class Balancing**: Addressed a significant 85/15 imbalance to ensure the "FAIL" class is accurately detected.

## 🛤️ Strategic Roadmap
The project is currently in **Phase 2 (Insights & Modularization)**. 
Future planned phases include:
- **Phase 3**: Persistence layer (Database integration to track student progress over time).
- **Phase 4**: Security (User Authentication & Role-Based Access Control).

---
*Created as part of the FIP 2026 Capstone Project.*

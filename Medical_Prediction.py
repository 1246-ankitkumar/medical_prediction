import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. UI & Theme Configuration ---
st.set_page_config(page_title="MediCost AI Pro", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #161b22; border: 1px solid #4facfe; border-radius: 10px; padding: 15px; }
    .stButton>button { background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); color: black; font-weight: bold; border-radius: 20px; transition: 0.3s; width: 100%; }
    .stButton>button:hover { transform: scale(1.02); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Navigation Sidebar ---
with st.sidebar:
    st.header("🛡️ MediCost AI")
    st.markdown("---")
    menu = st.radio("Pipeline Stages", ["📥 Data Injection", "🛠️ Preprocessing Unit", "📈 Visual Insights", "🤖 Model Arena"])
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file:
    # Persistent Data Initialization
    if 'main_df' not in st.session_state:
        st.session_state['main_df'] = pd.read_csv(uploaded_file)
    
    df = st.session_state['main_df']
    
    # --- STAGE 1: DATA INJECTION ---
    if menu == "📥 Data Injection":
        st.title("📥 Dataset Injection")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Null Values", df.isna().sum().sum())
        
        st.subheader("Data Preview")
        st.dataframe(df.head(15), use_container_width=True)

    # --- STAGE 2: PREPROCESSING UNIT ---
    elif menu == "🛠️ Preprocessing Unit":
        st.title("🛠️ Data Cleaning & Refinement")
        
        tab_null, tab_outlier, tab_eng = st.tabs(["🧽 Missing Values", "📉 Outlier Removal", "✨ Feature Engineering"])
        
        with tab_null:
            st.subheader("Smart Fill System")
            if st.button("🚀 Handle All Missing Values"):
                new_df = df.copy()
                for col in new_df.columns:
                    # FIX: Explicitly check for numeric types to avoid 'median' on strings error
                    if pd.api.types.is_numeric_dtype(new_df[col]):
                        new_df[col] = new_df[col].fillna(new_df[col].median())
                    else:
                        # Fallback to mode for categorical/string data
                        mode_val = new_df[col].mode()
                        if not mode_val.empty:
                            new_df[col] = new_df[col].fillna(mode_val[0])
                
                st.session_state['main_df'] = new_df
                st.success("Successfully applied Median/Mode imputation!")
                st.rerun()

        with tab_outlier:
            st.subheader("IQR-Based Outlier Removal")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if num_cols:
                target_col = st.selectbox("Select column to clean", num_cols)
                fig_box = px.box(df, y=target_col, color_discrete_sequence=['#4facfe'])
                st.plotly_chart(fig_box, use_container_width=True)
                
                if st.button("🗑️ Prune Outliers"):
                    Q1, Q3 = df[target_col].quantile(0.25), df[target_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    
                    before = df.shape[0]
                    new_df = df[(df[target_col] >= lower) & (df[target_col] <= upper)].copy()
                    st.session_state['main_df'] = new_df
                    st.warning(f"Removed {before - new_df.shape[0]} rows containing outliers.")
                    st.rerun()
            else:
                st.info("No numerical columns available for outlier removal.")

        with tab_eng:
            st.subheader("Final Encoding")
            if st.button("🔧 Process Features for Machine Learning"):
                df_ml = df.copy()
                # Automated Risk Score Logic
                if 'smoker' in df_ml.columns and 'age' in df_ml.columns:
                    le = LabelEncoder()
                    # Ensure column is string before encoding to avoid mixed-type errors
                    df_ml['smoker_num'] = le.fit_transform(df_ml['smoker'].astype(str))
                    df_ml['risk_score'] = df_ml['age'] * df_ml['smoker_num']
                
                # One-Hot Encoding for all non-numeric columns
                df_ml = pd.get_dummies(df_ml, drop_first=True)
                st.session_state['processed_df'] = df_ml
                st.success("Features encoded! Model training is now unlocked.")
                st.dataframe(df_ml.head())

    # --- STAGE 3: VISUAL INSIGHTS ---
    elif menu == "📈 Visual Insights":
        st.title("📈 Visual Insights")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(num_cols) >= 1:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                target_v = st.selectbox("Distribution Analysis", num_cols)
                fig_h = px.histogram(df, x=target_v, marginal="violin", color_discrete_sequence=['#00f2fe'])
                st.plotly_chart(fig_h, use_container_width=True)
                
            with col_v2:
                st.subheader("Feature Correlation")
                corr = df[num_cols].corr()
                fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='Blues')
                st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Please upload data with numerical values for visualization.")

    # --- STAGE 4: MODEL ARENA ---
    elif menu == "🤖 Model Arena":
        st.title("🤖 Model Arena")
        
        if 'processed_df' not in st.session_state:
            st.error("⚠️ Pipeline Blocked: Please go to 'Preprocessing Unit' -> 'Feature Engineering' first.")
        else:
            data = st.session_state['processed_df']
            # Default to the last column as target, typical for most datasets
            target_y = st.selectbox("Select Target Variable (Y)", data.columns, index=len(data.columns)-1)
            
            X = data.drop(columns=[target_y])
            y = data[target_y]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Prepare Scaled Data for Distance-Based Models (KNN)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if st.button("🏆 Start Performance Duel"):
                models = {
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Decision Tree": DecisionTreeRegressor(random_state=42),
                    "K-Neighbors (K-Scan)": KNeighborsRegressor(n_neighbors=5)
                }
                
                results = []
                bar = st.progress(0)
                
                for i, (name, model) in enumerate(models.items()):
                    # Logic: Use scaled features for KNN, raw features for Trees
                    xt, xv = (X_train_s, X_test_s) if "K-Neighbors" in name else (X_train, X_test)
                    
                    model.fit(xt, y_train)
                    preds = model.predict(xv)
                    
                    results.append({
                        "Model": name, 
                        "R2 Score": round(r2_score(y_test, preds), 4), 
                        "MAE": round(mean_absolute_error(y_test, preds), 2)
                    })
                    bar.progress((i + 1) / len(models))
                
                res_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
                st.table(res_df)
                st.balloons()
                st.success(f"Champion: {res_df.iloc[0]['Model']}")

else:
    st.info("👋 Welcome! Please upload your dataset to launch the MediCost pipeline.")
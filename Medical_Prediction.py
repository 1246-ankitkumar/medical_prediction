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
from sklearn.feature_selection import SelectKBest, f_regression

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="MediCost AI Pro", layout="wide")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🛡️ MediCost AI Pro")
    menu = st.radio("Pipeline", ["📥 Data", "🛠️ Preprocessing", "📈 Visuals", "🤖 Model"])
    file = st.file_uploader("Upload CSV")

# ---------------- LOAD DATA ----------------
if file:

    if 'main_df' not in st.session_state:
        st.session_state['main_df'] = pd.read_csv(file)

    df = st.session_state['main_df']

    # ================= DATA =================
    if menu == "📥 Data":
        st.title("📥 Dataset Overview")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
        st.write("Missing Values:", df.isna().sum().sum())

    # ================= PREPROCESS =================
    elif menu == "🛠️ Preprocessing":
        st.title("🛠️ Data Preprocessing")

        tab1, tab2, tab3 = st.tabs(["🧽 Missing Values", "📉 Outliers", "✨ Feature Engineering"])

        # -------- Missing Values --------
        with tab1:
            if st.button("Fix Missing Values"):
                new_df = df.copy()

                for col in new_df.columns:
                    if pd.api.types.is_numeric_dtype(new_df[col]):
                        if new_df[col].isna().all():
                            new_df[col] = new_df[col].fillna(0)
                        else:
                            new_df[col] = new_df[col].fillna(new_df[col].median())
                    else:
                        new_df[col] = new_df[col].fillna(new_df[col].mode()[0] if not new_df[col].mode().empty else "Unknown")

                st.session_state['main_df'] = new_df
                st.success("Missing values handled!")
                st.rerun()

        # -------- Outliers --------
        with tab2:
            st.subheader("Outlier Handling")

            num_cols = df.select_dtypes(include=np.number).columns.tolist()

            if num_cols:
                col = st.selectbox("Select Column", num_cols)

                fig = px.box(df, y=col)
                st.plotly_chart(fig, use_container_width=True)

                if st.button("Apply Capping"):
                    new_df = df.copy()

                    Q1 = new_df[col].quantile(0.25)
                    Q3 = new_df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR

                    new_df[col] = np.clip(new_df[col], lower, upper)

                    st.session_state['main_df'] = new_df
                    st.success("Outliers capped!")
                    st.rerun()

        # -------- Feature Engineering --------
        with tab3:
            if st.button("Encode Data"):
                df_ml = df.copy()

                if 'smoker' in df_ml.columns and 'age' in df_ml.columns:
                    le = LabelEncoder()
                    df_ml['smoker_num'] = le.fit_transform(df_ml['smoker'].astype(str))
                    df_ml['risk_score'] = df_ml['age'] * df_ml['smoker_num']

                df_ml = pd.get_dummies(df_ml, drop_first=True)
                df_ml = df_ml.astype(int)

                st.session_state['processed_df'] = df_ml
                st.success("Encoding Done")

            if 'processed_df' in st.session_state:
                data = st.session_state['processed_df']

                target = st.selectbox("Select Target Column", data.columns)

                if target not in data.columns:
                    st.error("Invalid target selected")
                    st.stop()

                X = data.drop(columns=[target])
                y = data[target]

                method = st.radio("Feature Selection", ["Correlation", "SelectKBest", "Feature Importance"])

                # -------- Correlation --------
                if method == "Correlation":
                    corr = data.corr(numeric_only=True)[target].abs().sort_values(ascending=False)
                    st.dataframe(corr)

                    k = st.slider("Top Features", 1, len(X.columns), 5)
                    selected = corr.index[1:k+1]

                # -------- SelectKBest --------
                elif method == "SelectKBest":
                    k = st.slider("K Features", 1, len(X.columns), 5)
                    selector = SelectKBest(f_regression, k=k)
                    selector.fit(X, y)
                    selected = X.columns[selector.get_support()]

                # -------- Feature Importance --------
                else:
                    model = RandomForestRegressor()
                    model.fit(X, y)

                    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    st.dataframe(imp)

                    k = st.slider("Top Features", 1, len(X.columns), 5)
                    selected = imp.head(k).index

                st.success(f"Selected Features: {list(selected)}")

                final_df = data[list(selected) + [target]]
                st.session_state['final_df'] = final_df

                st.dataframe(final_df.head())

    # ================= VISUALS =================
    elif menu == "📈 Visuals":
        st.title("📊 Simple Data Visualization")

        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) == 0:
            st.warning("No numeric columns found")
        else:

            feature = st.selectbox("Select Feature", num_cols)

            # Histogram
            st.subheader("Distribution")
            fig1 = px.histogram(df, x=feature, nbins=30)
            st.plotly_chart(fig1, use_container_width=True)

            # Boxplot
            st.subheader("Outliers (Boxplot)")
            fig2 = px.box(df, y=feature)
            st.plotly_chart(fig2, use_container_width=True)

            # Correlation
            st.subheader("Correlation Heatmap")
            corr = df[num_cols].corr()

            fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="Blues")
            st.plotly_chart(fig3, use_container_width=True)

    # ================= MODEL =================
    elif menu == "🤖 Model":
        st.title("🤖 Model Training")

        if 'processed_df' not in st.session_state:
            st.error("Run preprocessing first")
        else:
            data = st.session_state.get('final_df', st.session_state['processed_df'])

            target = st.selectbox("Select Target", data.columns)

            if target not in data.columns:
                st.error("Invalid target")
                st.stop()

            X = data.drop(columns=[target])
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if st.button("Train Models"):

                models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "KNN": KNeighborsRegressor()
                }

                results = []

                for name, model in models.items():

                    if name == "KNN":
                        xt, xv = X_train_s, X_test_s
                    else:
                        xt, xv = X_train, X_test

                    model.fit(xt, y_train)
                    pred = model.predict(xv)

                    results.append({
                        "Model": name,
                        "R2 Score": round(r2_score(y_test, pred), 4),
                        "MAE": round(mean_absolute_error(y_test, pred), 2)
                    })

                res_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

                st.dataframe(res_df)

                st.success(f"Best Model: {res_df.iloc[0]['Model']}")

else:
    st.info("Upload a CSV file to start the application")

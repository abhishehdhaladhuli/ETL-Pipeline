import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn import __version__ as sklearn_version
from packaging import version

# OneHotEncoder compatibility
if version.parse(sklearn_version) >= version.parse("1.2"):
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Load CSV or Excel
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        st.error("❌ Unsupported file format. Please upload a .csv or .xlsx file.")
        return None

# Drop columns with high missing values and duplicates
def drop_high_missing_and_duplicates(df, threshold=0.5):
    before_cols = set(df.columns)
    df = df.dropna(axis=1, thresh=int((1 - threshold) * len(df)))
    df = df.drop_duplicates()
    dropped_cols = before_cols - set(df.columns)
    return df, list(dropped_cols)

# Remove outliers from numeric columns
def remove_outliers(df, contamination=0.01):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df[numeric_cols])
    return df[preds == 1]

# Create preprocessing pipeline
def create_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', onehot)
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    return preprocessor

# Convert to downloadable CSV bytes
def convert_df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="ETL Preprocessing App", layout="wide")
st.title("🧪 ETL Preprocessing App (CSV & Excel)")

uploaded_file = st.file_uploader("📁 Upload your file", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.write("### 📊 Data Preview", df.head())
        st.write(f"✅ Shape: `{df.shape[0]}` rows × `{df.shape[1]}` columns")

        # Summary statistics
        st.write("### 📈 Summary Statistics")
        st.write(df.describe(include='all'))

        target_column = st.selectbox("🎯 Select target column", df.columns)
        contamination_rate = st.slider("📉 Outlier Contamination Rate", 0.0, 0.2, 0.01)

        if target_column:
            # Cleaning
            df, dropped_cols = drop_high_missing_and_duplicates(df)
            df = remove_outliers(df, contamination=contamination_rate)

            if target_column not in df.columns:
                st.error("❌ Selected target column not found.")
            else:
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Encode target if categorical
                if y.dtype == 'object' or y.dtype.name == 'category':
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                # Split BEFORE fitting preprocessor
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                preprocessor = create_preprocessor(X_train)

                # Fit on training, transform both sets
                X_train_transformed = preprocessor.fit_transform(X_train)
                X_test_transformed = preprocessor.transform(X_test)

                # Clean column names
                raw_cols = preprocessor.get_feature_names_out()
                transformed_cols = [col.split("__")[-1] for col in raw_cols]

                X_train_df = pd.DataFrame(X_train_transformed, columns=transformed_cols)
                X_test_df = pd.DataFrame(X_test_transformed, columns=transformed_cols)

                st.success("✅ ETL & Preprocessing Completed without data leakage!")

                if dropped_cols:
                    st.warning(f"⚠️ Dropped columns due to high missing values: {', '.join(dropped_cols)}")

                # Download buttons
                st.download_button("⬇️ Download X_train.csv", convert_df_to_csv_bytes(X_train_df), "X_train.csv")
                st.download_button("⬇️ Download X_test.csv", convert_df_to_csv_bytes(X_test_df), "X_test.csv")
                st.download_button("⬇️ Download y_train.csv", convert_df_to_csv_bytes(pd.DataFrame(y_train)), "y_train.csv")
                st.download_button("⬇️ Download y_test.csv", convert_df_to_csv_bytes(pd.DataFrame(y_test)), "y_test.csv")

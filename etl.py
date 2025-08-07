import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

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
    df = df.dropna(axis=1, thresh=int((1 - threshold) * len(df)))
    df = df.drop_duplicates()
    return df

# Remove outliers from numeric columns
def remove_outliers(df, contamination=0.01):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df[numeric_cols])
    return df[preds == 1]

# Preprocess: Missing values, scaling, encoding
def preprocess_data(df, target_column):
    df = drop_high_missing_and_duplicates(df)
    df = remove_outliers(df)

    if target_column not in df.columns:
        st.error("❌ Selected target column not found.")
        return None, None, None, None

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    return preprocessor, X, y, df.columns.difference(X.columns).tolist()  # also return dropped cols

# Transform features
def transform_data(preprocessor, X):
    return preprocessor.fit_transform(X)

# Train-test split
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

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

        target_column = st.selectbox("🎯 Select target column", df.columns)

        if target_column:
            preprocessor, X, y, dropped_cols = preprocess_data(df, target_column)

            if preprocessor is not None:
                X_transformed = transform_data(preprocessor, X)
                X_train, X_test, y_train, y_test = split_data(X_transformed, y)

                st.success("✅ ETL & Preprocessing Completed!")

                if dropped_cols:
                    st.warning(f"⚠️ Dropped columns: {', '.join(dropped_cols)}")

                # Download buttons
                st.download_button("⬇️ Download X_train.csv", convert_df_to_csv_bytes(pd.DataFrame(X_train)), "X_train.csv")
                st.download_button("⬇️ Download X_test.csv", convert_df_to_csv_bytes(pd.DataFrame(X_test)), "X_test.csv")
                st.download_button("⬇️ Download y_train.csv", convert_df_to_csv_bytes(pd.DataFrame(y_train)), "y_train.csv")
                st.download_button("⬇️ Download y_test.csv", convert_df_to_csv_bytes(pd.DataFrame(y_test)), "y_test.csv")

# ETL Preprocessing App

## 📊 Overview

This Streamlit-based ETL (Extract, Transform, Load) application allows users to upload CSV or Excel datasets and automatically processes them by:

* Handling missing values
* Removing outliers
* Encoding categorical variables
* Scaling numeric features
* Splitting into train-test sets

Users can then download the preprocessed datasets for machine learning tasks.

---

## 🧱 Technologies Used

* **Streamlit**: For building the interactive web app.
* **Pandas**: For reading and manipulating tabular data.
* **NumPy**: For efficient numeric operations.
* **Scikit-learn**:

  * `train_test_split` for splitting data
  * `SimpleImputer`, `StandardScaler`, `OneHotEncoder` for preprocessing
  * `Pipeline` and `ColumnTransformer` to streamline transformations
  * `IsolationForest` for outlier detection

---

## 🔧 Code Explanation

### 1. **File Loading**

```python
def load_data(uploaded_file):
```

Loads either `.csv` or `.xlsx` file using pandas. If unsupported, throws an error.

### 2. **Drop Columns and Duplicates**

```python
def drop_high_missing_and_duplicates(df, threshold=0.5):
```

Removes columns with more than 50% missing values and drops duplicate rows.

### 3. **Outlier Removal**

```python
def remove_outliers(df, contamination=0.01):
```

Uses `IsolationForest` to remove rows that are statistically anomalous in numeric columns.

### 4. **Preprocessing Function**

```python
def preprocess_data(df, target_column):
```

* Splits the dataframe into `X` and `y`
* Identifies numeric and categorical columns
* Creates preprocessing pipelines:

  * **Numeric**: Impute missing values with mean, scale using StandardScaler
  * **Categorical**: Impute with mode, encode with OneHotEncoder
* Returns the combined `ColumnTransformer` and other relevant objects.

### 5. **Transformation**

```python
def transform_data(preprocessor, X):
```

Fits and transforms `X` using the `preprocessor` pipeline.

### 6. **Train-Test Split**

```python
def split_data(X, y):
```

Splits features and target into training and testing sets using an 80/20 ratio.

### 7. **CSV Export**

```python
def convert_df_to_csv_bytes(df):
```

Converts any DataFrame to a downloadable CSV byte object.

---

## 🖊️ Streamlit UI Workflow

1. **Page Config and Title**

```python
st.set_page_config()
st.title()
```

Sets the web page layout and displays app title.

2. **File Uploader**

```python
uploaded_file = st.file_uploader()
```

Accepts CSV or Excel file from user.

3. **Target Column Selection**

```python
st.selectbox("Select target column")
```

User selects the column to be predicted (e.g. for supervised learning).

4. **Data Preview and Stats**

```python
st.write("Data Preview", df.head())
```

Shows the top 5 rows and shape of uploaded dataset.

5. **Preprocessing**

```python
preprocessor, X, y, dropped_cols = preprocess_data(...)
```

Cleans and transforms the dataset.

6. **Splitting and Transformation**

```python
X_train, X_test, y_train, y_test = split_data(...)
```

Performs train-test split after transformations.

7. **Download Buttons**

```python
st.download_button("Download X_train.csv"...)
```

Allows users to download all the processed datasets.

---

## 🚀 How to Run the App

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Streamlit app:

```bash
streamlit run etl.py  # or streamlit_app.py
```

---

## 🌐 Live Demo

Deployed using Streamlit Community Cloud:

> [ETL Preprocessing App](https://etl-pipeline-kpsftctfu3cynqyt9uh4te.streamlit.app/)

---


## 🙏 Acknowledgements

Special thanks to open-source libraries like Streamlit and Scikit-learn that made this possible!

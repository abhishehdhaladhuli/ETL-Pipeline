# CodeTech-DataScience-Task-1-ETL-Pipeline Using Scikitlearn and Pandas 
COMPANY: CODETECH IT SOLUTIONS
NAME: ABHISHEK DHALADHULI
INTERN ID: CT6WVEQ
DOMAIN: DATA SCIENCE
DURATION:6 WEEKS
MENTOR:NEELA SANTHOSH
# DESCRIPTION OF THE TASK-1 WHICH  IS BUILDING ETL PIPELINE USING SCIKITLEARN AND PANDAS 

The provided Python script is a comprehensive example of an ETL (Extract, Transform, Load) pipeline that automates the process of preprocessing, transforming, and loading data. This script leverages widely used libraries such as Pandas and Scikit-learn to handle various stages of the ETL process. Below is an in-depth description of the script, including the technologies used, the procedure, and the rationale behind each step.

## Technologies Used

1. **Pandas**:
    - A powerful data manipulation and analysis library for Python.
    - It provides data structures like DataFrame, which is essential for handling tabular data.
    - Used for loading data from CSV files and saving processed data back to CSV format.

2. **Scikit-learn**:
    - A robust machine learning library in Python.
    - Provides tools for data preprocessing, model building, and evaluation.
    - Key components used in the script include:
        - `train_test_split` for splitting data into training and testing sets.
        - `StandardScaler` and `OneHotEncoder` for data transformation.
        - `ColumnTransformer` for bundling multiple preprocessing steps.
        - `Pipeline` for creating sequential preprocessing steps.
        - `SimpleImputer` for handling missing values.

## Procedure

### 1. Load Data
The script begins by defining a function `load_data` that takes a file path as input and uses Pandas to load the data from a CSV file into a DataFrame.

```python
def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)
```

### 2. Preprocess Data
The `preprocess_data` function is responsible for preparing the data for analysis. It performs the following tasks:

- **Separates Features and Target**:
    - The target column is identified and separated from the feature columns.
    - This step is crucial for supervised learning tasks where the target variable is predicted based on the features.

```python
def preprocess_data(df, target_column):
    """Preprocess the dataset."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
```

- **Identifies Numerical and Categorical Columns**:
    - The script identifies columns with numerical data types (`int64` and `float64`) and categorical data types (`object`).

```python
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
```

- **Defines Preprocessing Steps**:
    - For numerical data, it creates a pipeline that imputes missing values using the mean and scales the data using `StandardScaler`.

```python
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
```

    - For categorical data, it creates a pipeline that imputes missing values using the most frequent strategy and encodes the data using `OneHotEncoder`.

```python
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
```

- **Combines Preprocessing Steps**:
    - The numerical and categorical preprocessing steps are combined using `ColumnTransformer`.

```python
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
```

- **Returns Preprocessing Pipeline and Data**:
    - The function returns the preprocessing pipeline, the feature DataFrame (`X`), and the target Series (`y`).

```python
    return preprocessor, X, y
```

### 3. Transform Data
The `transform_data` function applies the preprocessing pipeline to the feature DataFrame (`X`) to transform the data.

```python
def transform_data(preprocessor, X):
    """Transform the dataset using the preprocessor."""
    return preprocessor.fit_transform(X)
```

### 4. Split Data
The `split_data` function splits the transformed data into training and testing sets using Scikit-learn's `train_test_split`.

```python
def split_data(X, y):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Main Function
The `main` function orchestrates the entire ETL process. It performs the following steps:

- **Requests User Input**:
    - Prompts the user to input the dataset file path, target column name, and output path for saving the processed data.

```python
def main():
    file_path = input("Enter the path to the dataset file (e.g., data/dataset.csv): ")
    target_column = input("Enter the name of the target column: ")
    output_path = input("Enter the output path to save the processed data (e.g., data/): ")
```

- **Loads the Dataset**:
    - Calls the `load_data` function to read the dataset into a DataFrame.

```python
    df = load_data(file_path)
```

- **Preprocesses the Data**:
    - Calls `preprocess_data` to create the preprocessing pipeline and separates the features and target.

```python
    preprocessor, X, y = preprocess_data(df, target_column)
```

- **Transforms the Data**:
    - Applies the preprocessing pipeline to the features using `transform_data`.

```python
    X_transformed = transform_data(preprocessor, X)
```

- **Splits the Data**:
    - Splits the transformed data into training and testing sets using `split_data`.

```python
    X_train, X_test, y_train, y_test = split_data(X_transformed, y)
```

- **Saves the Processed Data**:
    - Saves the training and testing sets to CSV files at the specified output path.

```python
    pd.DataFrame(X_train).to_csv(output_path + 'X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv(output_path + 'X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv(output_path + 'y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv(output_path + 'y_test.csv', index=False)
```

- **Prints Completion Message**:
    - Informs the user that the ETL process has been completed successfully.

```python
    print("ETL process completed successfully!")
```

### Execution
The script is executed by calling the `main` function when the script is run directly.

```python
if __name__ == "__main__":
    main()
```

## Summary
This script provides a dynamic and flexible ETL pipeline for preprocessing, transforming, and loading data. By leveraging Pandas and Scikit-learn, it handles various preprocessing tasks such as imputing missing values, scaling numerical features, and encoding categorical features. The user-friendly interface allows users to input the dataset file path, target column name, and output path, making the script adaptable to different datasets and use cases. This script serves as a robust foundation for more complex data processing pipelines and can be extended to include additional preprocessing steps or transformations as needed.

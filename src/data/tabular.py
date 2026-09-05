import torch
import numpy as np
import pandas as pd
from torch.utils.data import (
    DataLoader,
    TensorDataset
)
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def remove_duplicate_rows(X, y):
    combined = X.copy()
    combined["target"] = y

    duplicate_mask = combined.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count > 0:
        keep_mask = ~duplicate_mask
        X = X.loc[keep_mask].copy()
        y = y.loc[keep_mask].copy()

    return X,y, duplicate_count


def remove_missing_label(X, y):
    missing_label_mask = y.isna()
    missing_label_count = int(missing_label_mask.sum())
    X = X.loc[~missing_label_mask].copy()
    y = y.loc[~missing_label_mask].copy()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y, missing_label_count


def remove_identifier_columns(X, identifier_columns):
    missing_identifier_columns = (
        set(identifier_columns) - set(X.columns)
    )
    if missing_identifier_columns:
        raise ValueError(
            "Specified identifier columns were not found: "
            f"{sorted(missing_identifier_columns)}"
        )

    X = X.drop(columns=identifier_columns)
    if X.shape[1] == 0:
        raise ValueError(
            "No predictor columns remain after column removal."
        )
    return X


def remove_constant_columns(X):
    constant_columns = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
    X = X.drop(columns = constant_columns)
    if X.shape[1] == 0:
        raise ValueError(
            "No predictor columns remain after column removal."
        )
    return X, constant_columns


def get_num_cat_columns(X, categorical_columns=None, numeric_columns=None):
    X = X.copy()
    categorical_columns = list(categorical_columns or [])
    numeric_columns = list(numeric_columns or [])


    specified_columns = set(
        categorical_columns + numeric_columns
    )

    missing_columns = specified_columns - set(X.columns)

    if missing_columns:
        raise ValueError(
            "The following specified columns were not found: "
            f"{sorted(missing_columns)}"
        )

    overlapping_columns = (
        set(categorical_columns) & set(numeric_columns)
    )

    if overlapping_columns:
        raise ValueError(
            "The following columns were specified as both "
            "numerical and categorical: "
            f"{sorted(overlapping_columns)}"
        )

    for column in numeric_columns:
        original_non_missing = int(X[column].notna().sum())

        converted = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        converted_non_missing = int(converted.notna().sum())

        newly_missing = (
            original_non_missing - converted_non_missing
        )

        if newly_missing > 0:
            raise ValueError(
                f"Column '{column}' was specified as numerical, "
                f"but {newly_missing} non-missing values could not "
                "be converted to numbers."
            )

        X[column] = converted


    for column in categorical_columns:
        X[column] = X[column].astype("category")


    manually_specified = set(
        categorical_columns + numeric_columns
    )

    remaining_columns = [
        column
        for column in X.columns
        if column not in manually_specified
    ]

    automatically_numeric = (
        X[remaining_columns]
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    automatically_categorical = (
        X[remaining_columns]
        .select_dtypes(
            include=["object", "category", "string", "bool"]
        )
        .columns
        .tolist()
    )


    final_numeric_columns = (
        numeric_columns + automatically_numeric
    )

    final_categorical_columns = (
        categorical_columns + automatically_categorical
    )

    final_numeric_columns = list(
        dict.fromkeys(final_numeric_columns)
    )

    final_categorical_columns = list(
        dict.fromkeys(final_categorical_columns)
    )

    classified_columns = set(
        final_numeric_columns + final_categorical_columns
    )

    unclassified_columns = (
        set(X.columns) - classified_columns
    )

    if unclassified_columns:
        unclassified_types = {
            column: str(X[column].dtype)
            for column in sorted(unclassified_columns)
        }

        raise ValueError(
            "The following columns could not be classified as "
            "numerical or categorical: "
            f"{unclassified_types}"
        )

    return (
        X,
        final_numeric_columns,
        final_categorical_columns
    )


def split_data(X, y, test_size=0.20, validation_size=0.20,random_state=42):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str)).astype(np.int64)
    temporary_size = validation_size + test_size
    X_train, X_temporary, y_train, y_temporary = (
        train_test_split(
            X,
            y_encoded,
            test_size=temporary_size,
            random_state=random_state,
            stratify=y_encoded
        )
    )
    relative_test_size = test_size / temporary_size
    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temporary,
            y_temporary,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=y_temporary
        )
    )

    return X_train, X_test, X_validation, y_train, y_test, y_validation, label_encoder


def preprocess_features( X_train, X_validation, X_test, numeric_columns, categorical_columns):
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )
    if not transformers:
        raise ValueError(
            "No numerical or categorical columns were provided"
        )
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_validation_processed = preprocessor.transform(X_validation)
    X_test_processed = preprocessor.transform(X_test)

    return (
        np.asarray(X_train_processed, dtype=np.float32),
        np.asarray(X_validation_processed, dtype=np.float32),
        np.asarray(X_test_processed, dtype=np.float32),
    ) 


def create_dataloader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.as_tensor(X, dtype = torch.float32),
        torch.as_tensor(y, dtype = torch.long),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
import json
from collections import Counter

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.sparse
import seaborn as sns
import shap
import statsmodels.api as sm
from flask import Flask, jsonify, request
from imblearn.over_sampling import SMOTE
import math
from imblearn.pipeline import Pipeline as ImbPipeline
from IPython.display import Image, display
from scipy import stats
from scipy.stats import mode, randint, uniform
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.formula.api import logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

def full_eda_visualization(data, target_column):
    """
    Creates:
    - Count plots for categorical and binary columns
    - Violin plots and bar plots with confidence intervals for continuous columns
    """

    # Identify columns
    cat_columns = [
        col
        for col in data.select_dtypes(include=["object", "category", "bool"]).columns
        if data[col].nunique(dropna=True) <= 20 and col != target_column
    ]

    num_columns = [
        col
        for col in data.select_dtypes(include=["int64", "float64"]).columns
        if col != target_column
    ]

    binary_cols = [col for col in num_columns if data[col].nunique(dropna=True) == 2]
    continuous_cols = [col for col in num_columns if data[col].nunique(dropna=True) > 2]

    combined_columns = cat_columns + binary_cols

    # --- Count plots ---
    n_cols = 5
    n_rows = (len(combined_columns) // n_cols) + int(
        len(combined_columns) % n_cols != 0
    )
    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.6))
        axes = axes.flatten()

        for i, col in enumerate(combined_columns):
            ax = axes[i]
            sns.countplot(x=col, hue=target_column, data=data, ax=ax)
            ax.set_title(f"{col} by {target_column}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(
                        f"{int(height)}",
                        (p.get_x() + p.get_width() / 2, height),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="black",
                    )

        for j in range(len(combined_columns), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        plt.show()

    # --- Violin plots for continuous columns ---
    n_cols = 5
    n_rows = (len(continuous_cols) // n_cols) + int(len(continuous_cols) % n_cols != 0)
    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.6))
        axes = axes.flatten()

        for i, col in enumerate(continuous_cols):
            ax = axes[i]
            sns.violinplot(x=target_column, y=col, data=data, ax=ax, inner="point")
            ax.set_title(f"{col} by {target_column}")
            ax.set_xlabel(target_column)
            ax.set_ylabel(col)

            counts = data.groupby(target_column)[col].count()
            for cat, count in counts.items():
                ax.text(
                    cat,
                    data[col].max(),
                    f"n={count}",
                    horizontalalignment="center",
                    color="black",
                    fontsize=9,
                    weight="bold",
                )

        for j in range(len(continuous_cols), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    # --- Bar plots (mean + 95% CI) for continuous columns ---
    n_cols = 7
    n_rows = (len(continuous_cols) // n_cols) + int(len(continuous_cols) % n_cols != 0)
    if n_rows > 0:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 3.0)
        ) 
        axes = axes.flatten()

        for i, col in enumerate(continuous_cols):
            ax = axes[i]
            sns.barplot(x=target_column, y=col, data=data, errorbar=("ci", 95), ax=ax)
            ax.set_title(f"Mean {col} by {target_column} with 95% CI", fontsize=10)
            ax.set_xlabel(target_column, fontsize=9)
            ax.set_ylabel(f"Mean {col}", fontsize=9)

            # Calculate mean per target group for current col
            means = data.groupby(target_column)[col].mean()

            max_mean = means.max()
            margin = max_mean * 0.10 if max_mean > 0 else 0.2  # Increased margin here!

            ax.set_ylim(bottom=0, top=max_mean + margin)

            # Show value of each bar on top of the bar
            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height):
                    ax.annotate(
                        f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2, height),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="black",
                    )

        for j in range(len(continuous_cols), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.6, wspace=0.3)
        plt.show()

def print_column_descriptions(columns_info_df, table_name):
    """
    Prints cleaned, readable descriptions of columns
    for a given dataset/table.
    
    Args:
        column_info_df (pd.DataFrame): Data dictionary with 'Table', 'Row', 'Description'
        table_name (str): Name of the table to filter (e.g., 'bureau.csv')
    """
    filtered = columns_info_df[columns_info_df["Table"] == table_name]

    print(f"\n📄 Column Descriptions for '{table_name}':\n")
    for _, row in filtered.iterrows():
        col_name = row["Row"]
        desc = row["Description"]
        print(f"- {col_name}: {desc}")

def plot_target_distribution(df, target_column):
    """
    Plots a bar chart showing the count of each unique value in the target column.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df, palette="Set2")
    plt.title(f"Distribution of {target_column}")
    plt.xlabel(target_column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def is_binary_column(column):
    """
    Check if a pandas Series (column) contains only binary values (0 and 1).

    Parameters:
    -----------
    column : pandas.Series
        The column to check.

    Returns:
    --------
    bool
        True if the column contains only 0s and 1s (ignoring NaNs), else False.
    """
    unique_values = column.dropna().unique()
    return set(unique_values).issubset({0, 1})


def get_column_aggregations(df):
    """
    Generate a dictionary of aggregation functions for each column in a DataFrame.

    Binary columns will be aggregated by mode, numeric columns by mean, and other
    columns by mode.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame whose columns will be assigned aggregation functions.

    Returns:
    --------
    dict
        A dictionary mapping column names to aggregation functions or strings.
    """
    aggregations = {}
    for column in df.columns:
        if is_binary_column(df[column]):
            aggregations[column] = (
                lambda x: x.mode(dropna=True).iloc[0] if not x.mode(dropna=True).empty else None
            )
        elif pd.api.types.is_numeric_dtype(df[column]):
            aggregations[column] = 'mean'
        else:
            aggregations[column] = (
                lambda x: x.mode(dropna=True).iloc[0] if not x.mode(dropna=True).empty else None
            )
    return aggregations


def merge_tables_in_chunks(main_df, sub_df, main_key, sub_key, sub_df_name='sub', chunk_size=100000):
    """
    Merge two DataFrames by aggregating the sub_df in chunks and joining to main_df.

    The sub_df is aggregated by sub_key using column-specific aggregation functions.
    The aggregated sub_df is then merged into main_df on their respective keys.

    Parameters:
    -----------
    main_df : pandas.DataFrame
        The main DataFrame to merge into.

    sub_df : pandas.DataFrame
        The DataFrame to aggregate and merge from.

    main_key : str
        The key column name in main_df to join on.

    sub_key : str
        The key column name in sub_df to join on.

    sub_df_name : str, optional (default='sub')
        A suffix to add to overlapping columns from sub_df after merge.

    chunk_size : int, optional (default=100000)
        The number of unique keys to process in each chunk.

    Returns:
    --------
    pandas.DataFrame
        The merged DataFrame with aggregated sub_df data appended.
    """
    # Determine aggregation functions for sub_df columns
    aggregations = get_column_aggregations(sub_df)

    # Remove the key from aggregations (we don't want to aggregate on it)
    if sub_key in aggregations:
        del aggregations[sub_key]

    # Index both DataFrames by their key
    main_df_indexed = main_df.set_index(main_key)
    sub_df_indexed = sub_df.set_index(sub_key)

    # Get unique keys
    unique_keys = sub_df_indexed.index.dropna().unique()
    total_keys = len(unique_keys)

    aggregated_sub_chunks = []

    # Aggregate sub_df in chunks
    for start in range(0, total_keys, chunk_size):
        end = start + chunk_size
        chunk_keys = unique_keys[start:end]

        chunk = sub_df_indexed.loc[chunk_keys]
        aggregated_chunk = chunk.groupby(level=0).agg(aggregations)
        aggregated_sub_chunks.append(aggregated_chunk)

    # Combine and aggregate again (to catch duplicates across chunks)
    aggregated_sub = pd.concat(aggregated_sub_chunks)
    aggregated_sub = aggregated_sub.groupby(aggregated_sub.index).agg(aggregations)

    # Merge with main_df using a readable suffix
    result = main_df_indexed.merge(
        aggregated_sub,
        left_index=True,
        right_index=True,
        how='left',
        suffixes=('', f'_{sub_df_name}')
    ).reset_index()

    return result

def calculate_missing_data(df):
    """
    Calculates the percentage of missing values for each column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary with column names as keys and percentage of missing values as values.
    """
    missing_data = {}
    total_rows = len(df)

    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_percentage = (missing_count / total_rows) * 100
        missing_data[col] = round(missing_percentage, 2)

    return missing_data

def visualize_missing_data(missing_columns_only):
    """
    Visualizes missing data:
    1) Boxplot of all missing percentages.
    2) Bar chart of top 10 columns with most missing data.
    3) Bar chart of top 10 columns with least missing data.
    """
    # 1. Boxplot
    plt.figure(figsize=(8, 5))
    plt.boxplot(missing_columns_only.values(), vert=False)
    plt.title("Boxplot of Missing Data Percentages")
    plt.xlabel("Percentage")
    plt.tight_layout()
    plt.show()

    # 2. Top 10 columns with most missing data
    top_10 = dict(sorted(missing_columns_only.items(), key=lambda x: x[1], reverse=True)[:10])
    plt.figure(figsize=(10, 6))
    plt.bar(top_10.keys(), top_10.values(), color='tomato')
    plt.title("Top 10 Columns with Highest Missing Data")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("% Missing")
    plt.tight_layout()
    plt.show()

    # 3. Top 10 columns with least missing data
    bottom_10 = dict(sorted(missing_columns_only.items(), key=lambda x: x[1])[:10])
    plt.figure(figsize=(10, 6))
    plt.bar(bottom_10.keys(), bottom_10.values(), color='mediumseagreen')
    plt.title("Top 10 Columns with Lowest Missing Data")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("% Missing")
    plt.tight_layout()
    plt.show()

def identify_columns_to_drop(df):
    """
    Identifies columns to drop from the DataFrame based on:
    - Any column with >= 50% missing data.
    - Any non-numeric column with > 30% missing data.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: Column names that meet the drop criteria.
    """
    total_rows = len(df)
    columns_to_drop = []

    for col in df.columns:
        missing_pct = df[col].isna().sum() / total_rows * 100

        if missing_pct >= 50:
            columns_to_drop.append(col)
        elif not pd.api.types.is_numeric_dtype(df[col]) and missing_pct > 30:
            columns_to_drop.append(col)

    return columns_to_drop

def check_linear_logit_assumption(model_sig, df_sampled, target_col="TARGET"):
    sig_vars = model_sig.params.index.drop("Intercept")
    
    fail_count = 0
    pass_count = 0
    
    for predictor in sig_vars:
        # Add squared term
        df_sampled[f"{predictor}_squared"] = df_sampled[predictor] ** 2
        
        formula = f"{target_col} ~ {predictor} + {predictor}_squared"
        
        try:
            test_model = logit(formula, data=df_sampled).fit(disp=0)
        except Exception as e:
            print(f"Skipping {predictor} due to error: {e}")
            continue
        
        pval_sq = test_model.pvalues.get(f"{predictor}_squared", None)
        if pval_sq is not None and pval_sq < 0.05:
            fail_count += 1
        else:
            pass_count += 1
        
        # Optional: drop the squared column to keep df_sampled clean
        df_sampled.drop(columns=[f"{predictor}_squared"], inplace=True)
    
    print(f"Variables passing linear logit assumption: {pass_count}")
    print(f"Variables failing linear logit assumption: {fail_count}")

def impute_data_batched(df, n_neighbors=2, batch_size=10000):
    """
    Imputes missing values:
    - Continuous numeric columns: KNN imputation in batches
    - Categorical and binary columns: Mode imputation

    Args:
        df (pd.DataFrame): DataFrame with missing values
        n_neighbors (int): Number of neighbors for KNN
        batch_size (int): Number of rows per batch

    Returns:
        pd.DataFrame: DataFrame with imputed values
    """
    df_imputed = df.copy()

    # Separate numeric columns
    numeric_cols_all = df_imputed.select_dtypes(include=[np.number]).columns

    # Identify binary numeric columns (e.g., 0/1)
    binary_cols = [
        col for col in numeric_cols_all if df_imputed[col].nunique(dropna=True) == 2
    ]

    # True continuous numeric columns
    continuous_cols = [
        col for col in numeric_cols_all if col not in binary_cols
    ]

    # Treat binary as non-numeric (categorical) for imputation
    non_numeric_cols = (
        list(df_imputed.select_dtypes(exclude=[np.number]).columns) + binary_cols
    )

    # Mode Imputation for non-numeric + binary columns
    for col in non_numeric_cols:
        mode = df_imputed[col].mode(dropna=True)
        if not mode.empty:
            df_imputed[col].fillna(mode[0], inplace=True)

    # Batched KNN Imputation for continuous numeric columns
    if continuous_cols:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        n_rows = df_imputed.shape[0]
        result_chunks = []

        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            batch = df_imputed.iloc[start:end][continuous_cols]
            batch_imputed = imputer.fit_transform(batch)
            result_chunks.append(pd.DataFrame(batch_imputed, columns=continuous_cols, index=range(start, end)))

        # Combine imputed batches
        df_imputed[continuous_cols] = pd.concat(result_chunks)

    return df_imputed

def impute_data_median_mode(df):
    """
    Imputes missing values in a DataFrame.

    - Continuous numeric columns: Imputed using the median.
    - Categorical and binary columns: Imputed using the mode.

    Args:
        df (pd.DataFrame): The input DataFrame with potential missing values.

    Returns:
        pd.DataFrame: A new DataFrame where missing values have been imputed.
    """
    df_imputed = df.copy()

    numeric_cols_all = df_imputed.select_dtypes(include=[np.number]).columns
    binary_cols = [col for col in numeric_cols_all if df_imputed[col].nunique(dropna=True) == 2]
    continuous_cols = [col for col in numeric_cols_all if col not in binary_cols]
    non_numeric_cols = list(df_imputed.select_dtypes(exclude=[np.number]).columns) + binary_cols

    for col in continuous_cols:
        median = df_imputed[col].median(skipna=True)
        df_imputed[col] = df_imputed[col].fillna(median)

    for col in non_numeric_cols:
        mode = df_imputed[col].mode(dropna=True)
        if not mode.empty:
            df_imputed[col] = df_imputed[col].fillna(mode[0])

    return df_imputed

def winsorize_dataframe(df, limits=(0.025, 0.025)):
    """
    Applies 95% winsorization to all continuous numeric columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        limits (tuple): Lower and upper quantile limits (default is 2.5%)

    Returns:
        pd.DataFrame: Winsorized DataFrame
    """
    df_winsorized = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    binary_cols = [col for col in numeric_cols if df[col].nunique(dropna=True) == 2]
    continuous_cols = [col for col in numeric_cols if col not in binary_cols]

    for col in continuous_cols:
        df_winsorized[col] = winsorize(df[col], limits=limits)

    return df_winsorized

def robust_scale_dataframe(df):
    """
    Applies RobustScaler to continuous numeric columns.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Scaled DataFrame
    """
    df_scaled = df.copy()
    scaler = RobustScaler()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    binary_cols = [col for col in numeric_cols if df[col].nunique(dropna=True) == 2]
    continuous_cols = [col for col in numeric_cols if col not in binary_cols]

    df_scaled[continuous_cols] = scaler.fit_transform(df_scaled[continuous_cols])

    return df_scaled

def inspect_negative_value_distributions_with_descriptions(
    df, description_df,
    column_name_col='Row',
    description_col='Description',
    bins=50
):
    """
    Plots compact histograms of continuous numeric columns with negative values,
    and prints their metadata descriptions below each plot.

    Args:
        df (pd.DataFrame): The dataset to inspect.
        description_df (pd.DataFrame): DataFrame with metadata (e.g., HomeCredit_columns_description.csv).
        column_name_col (str): Column name in metadata with variable names.
        description_col (str): Column name in metadata with descriptions.
        bins (int): Number of bins for histograms.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    binary_cols = [col for col in numeric_cols if df[col].nunique(dropna=True) == 2]
    continuous_cols = [col for col in numeric_cols if col not in binary_cols]
    negative_cols = [col for col in continuous_cols if (df[col] < 0).any()]

    if not negative_cols:
        print("✅ No continuous columns with negative values.")
        return

    print(f"📉 Found {len(negative_cols)} continuous columns with negative values.\n")

    # 3 plots per row
    cols_per_row = 3
    n = len(negative_cols)
    rows = math.ceil(n / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 3 * rows))
    axes = axes.flatten()

    for idx, col in enumerate(negative_cols):
        ax = axes[idx]
        sns.histplot(df[col].dropna(), kde=True, bins=bins, ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Get description
        match = description_df[description_df[column_name_col] == col]
        description = match[description_col].values[0] if not match.empty else "⚠️ No description found."
        ax.annotate(description, xy=(0.5, -0.4), xycoords='axes fraction',
                    ha='center', fontsize=8, wrap=True)

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def encode_and_drop_high_cardinality(df, max_unique=15):
    """
    One-hot encode all non-numeric columns with unique values <= max_unique.
    Drop non-numeric columns with unique values > max_unique.

    Args:
        df (pd.DataFrame): Input DataFrame
        max_unique (int): Max unique values to keep and encode

    Returns:
        pd.DataFrame: DataFrame with encoded columns and high-cardinality cols dropped
    """
    df = df.copy()

    # Identify non-numeric columns
    cat_cols = df.select_dtypes(exclude=["number"]).columns

    # Columns to drop (high cardinality)
    drop_cols = [col for col in cat_cols if df[col].nunique(dropna=True) > max_unique]
    # Columns to one-hot encode (everything else non-numeric)
    encode_cols = [col for col in cat_cols if col not in drop_cols]

    print(f"🗑️ Dropping columns with > {max_unique} unique values: {drop_cols}")
    print(f"🔁 One-hot encoding columns: {encode_cols}")

    # Drop high-cardinality columns
    df = df.drop(columns=drop_cols)

    # One-hot encode the rest
    df_encoded = pd.get_dummies(df, columns=encode_cols, drop_first=True)

    return df_encoded

def convert_boolean_columns_to_int(df):
    """
    Convert columns containing only True/False values (bool or object dtype)
    to integer columns with 0/1.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with boolean columns converted to integers
    """
    bool_cols = []
    for col in df.columns:
        # Drop missing values and check if unique values are exactly {True, False} or subset thereof
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({True, False}):
            bool_cols.append(col)

    # Convert all detected boolean columns to int
    df[bool_cols] = df[bool_cols].astype(int)

    return df

def drop_highly_correlated_features(df, target_col=None, threshold=0.80):
    """
    Drops features from df that are highly correlated with each other (abs(corr) > threshold).
    If target_col is provided, it will be excluded from dropping.

    Args:
        df (pd.DataFrame): DataFrame including features and possibly a target column.
        target_col (str or None): Optional. Column to protect from being dropped (e.g., 'TARGET').
        threshold (float): Correlation threshold above which to consider features redundant.

    Returns:
        pd.DataFrame: DataFrame with reduced features.
        list: List of dropped feature column names.
    """
    # Separate target column if specified
    if target_col and target_col in df.columns:
        y = df[[target_col]]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()

    # Compute correlation matrix
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    X_reduced = X.drop(columns=to_drop)

    # Reattach target column if applicable
    if y is not None:
        df_reduced = pd.concat([X_reduced, y], axis=1)
    else:
        df_reduced = X_reduced

    return df_reduced, to_drop


def calculate_vif_and_drop_high_vif_iterative(
    df,
    target_col,
    sample_size=5000,
    large_sample_size=20000,
    random_state=42,
    vif_threshold=5,
    low_variance_threshold=1e-3  # increased threshold to catch more near-constant cols
):
    """
    Calculate Variance Inflation Factor (VIF) iteratively and drop high VIF or low-variance columns.
    Target column is retained throughout and reattached correctly at the end.

    Returns:
        - VIF DataFrame
        - Cleaned DataFrame (including target column)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    y = df[[target_col]]  # keep as DataFrame for concat later

    X = df.drop(columns=[target_col])

    # Drop low variance or constant cols on full data first (stricter threshold)
    low_var_cols_full = X.columns[X.var() < low_variance_threshold].tolist()
    if low_var_cols_full:
        print(f"Dropping low variance columns on full data before sampling: {low_var_cols_full}")
        X = X.drop(columns=low_var_cols_full)

    # Sample for iterative VIF dropping
    X_sampled = X.sample(n=sample_size, random_state=random_state) if len(X) > sample_size else X.copy()

    # Drop constant and near-constant columns on sampled data (more conservative threshold)
    nunique = X_sampled.nunique()
    constant_cols_sampled = nunique[nunique <= 1].index.tolist()
    low_variance_cols_sampled = X_sampled.columns[X_sampled.var() < low_variance_threshold].tolist()
    to_drop = list(set(constant_cols_sampled + low_variance_cols_sampled))
    if to_drop:
        print(f"Dropping constant/low variance columns on sampled data: {to_drop}")
        X_sampled = X_sampled.drop(columns=to_drop)

    dropped_cols = low_var_cols_full + to_drop

    # Iterative VIF removal
    while True:
        X_const = add_constant(X_sampled)
        vif = pd.Series(
            [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
            index=X_const.columns
        ).drop("const", errors="ignore")

        high_vif = vif[vif > vif_threshold]
        if high_vif.empty:
            break
        drop_var = high_vif.idxmax()
        print(f"Dropping '{drop_var}' with VIF={high_vif.max():.2f}")
        X_sampled = X_sampled.drop(columns=[drop_var])
        dropped_cols.append(drop_var)

    # Apply all drops to full data
    X_cleaned = X.drop(columns=dropped_cols, errors='ignore')

    # Large sample re-check for high VIF
    X_large = X_cleaned.copy()
    X_large_sampled = X_large.sample(n=large_sample_size, random_state=random_state) if len(X_large) > large_sample_size else X_large

    X_large_const = add_constant(X_large_sampled)
    large_vif_series = pd.Series(
        [variance_inflation_factor(X_large_const.values, i) for i in range(X_large_const.shape[1])],
        index=X_large_const.columns
    ).drop("const", errors="ignore")

    remaining_high_vif = large_vif_series[large_vif_series > vif_threshold]
    if not remaining_high_vif.empty:
        print(f"Dropping remaining high VIF columns on large sample: {remaining_high_vif.index.tolist()}")
        X_cleaned = X_cleaned.drop(columns=remaining_high_vif.index.tolist(), errors='ignore')

        # Recalculate final VIF on cleaned data
        final_sample = X_cleaned.sample(n=sample_size, random_state=random_state) if len(X_cleaned) > sample_size else X_cleaned
        final_vif = pd.DataFrame({
            "Variable": final_sample.columns,
            "VIF": [variance_inflation_factor(add_constant(final_sample).values, i+1) for i in range(final_sample.shape[1])]
        })
    else:
        final_vif = pd.DataFrame({
            "Variable": large_vif_series.index,
            "VIF": large_vif_series.values
        })

    # **Final check: drop any constant columns left in X_cleaned**
    const_final = X_cleaned.columns[X_cleaned.nunique() <= 1].tolist()
    if const_final:
        print(f"Dropping constant columns in final cleaned data: {const_final}")
        X_cleaned = X_cleaned.drop(columns=const_final)
        # Update final VIF accordingly (optional)

    # Reattach target column
    df_cleaned = pd.concat([X_cleaned, y.loc[X_cleaned.index]], axis=1)

    return final_vif, df_cleaned


def preprocess_and_fit_logit(df, target_col="TARGET", train_size=100000, random_state=42, near_constant_unique_threshold=2):
    # Copy and clean columns to valid names
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.replace(r'\W+', '_', regex=True)
    
    # Sample with stratification on the target
    df_sampled, _ = train_test_split(
        df_clean,
        train_size=train_size,
        stratify=df_clean[target_col],
        random_state=random_state
    )
    
    # Remove infinite and NaN values
    df_sampled = df_sampled.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Detect constant and near-constant columns excluding the target
    unique_counts = df_sampled.nunique()
    cols_to_drop = unique_counts[
        (unique_counts <= near_constant_unique_threshold) & (unique_counts.index != target_col)
    ].index.tolist()
    
    if cols_to_drop:
        df_sampled = df_sampled.drop(columns=cols_to_drop)
    
    # Prepare formula for logistic regression (full model)
    predictors = df_sampled.columns.drop(target_col)
    formula_full = f"{target_col} ~ " + " + ".join(predictors)
    
    # Fit logistic regression model (first model), suppress printing
    model_full = logit(formula_full, data=df_sampled).fit(disp=0)
    
    # Extract significant predictors with p-value < 0.05
    pvals = model_full.pvalues.drop("Intercept", errors="ignore")
    significant_vars = pvals[pvals < 0.05].index.tolist()
    
    # Fit second logistic regression with significant variables only
    if significant_vars:
        formula_sig = f"{target_col} ~ " + " + ".join(significant_vars)
        model_sig = logit(formula_sig, data=df_sampled).fit(disp=0)
        print("\nSummary of logistic regression model with significant variables only:")
        print(model_sig.summary())
        return model_sig, df_sampled
    else:
        print("No significant variables found. Returning full model.")
        return model_full, df_sampled

def evaluate_model_performance(y_test, y_pred, y_proba, class_labels=None):
    """
    Prints evaluation metrics and plots confusion matrix and ROC curve.

    Parameters:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted class labels.
        y_proba (array-like): Predicted probabilities for the positive class.
        class_labels (list): Labels for confusion matrix axes.
    """
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"\nTest AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")

    class_labels = list(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_learning_curve(
    model, X, y, scoring="average_precision", cv=5, title="Learning Curve"
):
    """
    Plots a learning curve showing training and validation scores.

    Parameters:
        model: Fitted estimator or pipeline.
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        scoring (str): Scoring metric to evaluate (default: "average_precision").
        cv (int or cross-validation generator): Number of CV folds or CV strategy.
        title (str): Title for the plot.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )

    plt.plot(train_sizes, val_mean, "o-", color="green", label="Validation score")
    plt.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="green"
    )

    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel(f"{scoring.replace('_', ' ').title()} Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def append_class_metrics(y_true, y_pred, pipeline, results_df):
    """
    Extract precision, recall, f1 for classes '0' and '1' from classification report,
    get the model name automatically from the pipeline,
    and append results to the given DataFrame safely without triggering FutureWarning.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    - pipeline: sklearn pipeline (used to extract model name)
    - results_df: pandas DataFrame to append results to

    Returns:
    - Updated results_df with the new row appended
    """
    # Get classification report as dict
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    # Extract metrics for class '0' safely (default 0.0 if missing)
    precision_0 = report_dict.get("0", {}).get("precision", 0.0)
    recall_0 = report_dict.get("0", {}).get("recall", 0.0)
    f1_0 = report_dict.get("0", {}).get("f1-score", 0.0)

    # Extract metrics for class '1' safely (default 0.0 if missing)
    precision_1 = report_dict.get("1", {}).get("precision", 0.0)
    recall_1 = report_dict.get("1", {}).get("recall", 0.0)
    f1_1 = report_dict.get("1", {}).get("f1-score", 0.0)

    # Attempt to extract a meaningful model name from pipeline
    model_name = "UnknownModel"
    try:
        if hasattr(pipeline, "named_steps"):
            clf = pipeline.named_steps.get("classifier", None)
            if clf is not None:
                model_name = type(clf).__name__
            else:
                first_step = list(pipeline.named_steps.values())[0]
                model_name = type(first_step).__name__
        else:
            model_name = type(pipeline).__name__
    except Exception:
        pass

    # Create new DataFrame row with metrics for both classes
    new_row = pd.DataFrame(
        {
            "model": [model_name],
            "precision_0": [precision_0],
            "recall_0": [recall_0],
            "f1_0": [f1_0],
            "precision_1": [precision_1],
            "recall_1": [recall_1],
            "f1_1": [f1_1],
        }
    )

    # Drop all-NA columns from both before concatenation to avoid FutureWarning
    results_df_clean = results_df.dropna(axis=1, how="all")
    new_row_clean = new_row.dropna(axis=1, how="all")

    # Concatenate safely
    results_df_updated = pd.concat([results_df_clean, new_row_clean], ignore_index=True)

    return results_df_updated

def get_feature_names(final_pipeline, cont_cols, cat_cols, bin_cols):
    """
    Extract final feature names from the full final_pipeline.

    Parameters:
        final_pipeline (Pipeline): Full pipeline with preprocessor and classifier.
        cont_cols, cat_cols, bin_cols: Lists of column names per type.

    Returns:
        np.ndarray of feature names.
    """
    cont_bin_cols = cont_cols + bin_cols

    # Step into the inner ColumnTransformer
    column_transformer = final_pipeline.named_steps["preprocessor"].named_steps["transform"]

    # Step into the 'cat' pipeline inside the ColumnTransformer
    cat_pipeline = column_transformer.named_transformers_["cat"]
    ohe = cat_pipeline.named_steps["onehot"]

    if not hasattr(ohe, "categories_"):
        raise ValueError("OneHotEncoder inside the pipeline is not fitted yet.")

    cat_features = ohe.get_feature_names_out(cat_cols)
    return np.concatenate([cont_bin_cols, cat_features])

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, missing_thresh_high=50, missing_thresh_low=30):
        self.missing_thresh_high = missing_thresh_high
        self.missing_thresh_low = missing_thresh_low

    def fit(self, X, y=None):
        X = X.copy()
        self.columns_to_drop_ = []

        for col in X.columns:
            missing_pct = X[col].isna().mean() * 100
            is_numeric = pd.api.types.is_numeric_dtype(X[col])
            unique_vals = X[col].dropna().unique()
            is_binary = set(unique_vals).issubset({0, 1})

            if missing_pct >= self.missing_thresh_high:
                self.columns_to_drop_.append(col)
            elif (
                not is_numeric or is_binary
            ) and missing_pct > self.missing_thresh_low:
                self.columns_to_drop_.append(col)

        self.binary_cols_ = [
            col
            for col in X.columns
            if col not in self.columns_to_drop_
            and X[col].nunique(dropna=True) == 2
            and pd.api.types.is_numeric_dtype(X[col])
        ]

        self.cat_cols_ = [
            col
            for col in X.columns
            if col not in self.columns_to_drop_
            and X[col].dtype == "object"
            and 2 < X[col].nunique() <= 10
        ]

        self.cont_cols_ = [
            col
            for col in X.columns
            if col not in self.columns_to_drop_
            and col not in self.binary_cols_
            and col not in self.cat_cols_
            and pd.api.types.is_numeric_dtype(X[col])
        ]
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors="ignore")

    def get_feature_names(self):
        return self.cont_cols_, self.cat_cols_, self.binary_cols_


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=(0.025, 0.025)):
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X, columns=getattr(X, "columns", range(X.shape[1])))
        for col in X_.columns:
            if X_[col].nunique() > 2:
                X_[col] = winsorize(X_[col], limits=self.limits)
        return X_


class TailClipper(BaseEstimator, TransformerMixin):
    def __init__(self, clip_percentile=0.99):
        self.clip_percentile = clip_percentile

    def fit(self, X, y=None):
        X_ = pd.DataFrame(X, columns=getattr(X, "columns", range(X.shape[1])))
        self.percentiles_ = X_.quantile(self.clip_percentile)
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X, columns=self.percentiles_.index)
        for col in self.percentiles_.index:
            X_[col] = np.clip(X_[col], a_min=None, a_max=self.percentiles_[col])
        return X_

class BalancedBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, preprocessor, base_estimator, random_state=None):
        """
        preprocessor    : a fitted Pipeline/Transformer (dropper→ColumnTransformer)
        base_estimator  : an unfitted classifier instance (e.g. LogisticRegression())
        """
        self.preprocessor = preprocessor
        self.base_estimator = base_estimator
        self.random_state = random_state

    def fit(self, X, y):
        # 1) make sure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = pd.Series(y, index=X.index)

        # 2) fit the preprocessor once on all data
        self.preprocessor_ = clone(self.preprocessor)
        self.preprocessor_.fit(X, y)
        # transform the *entire* train set once
        X_all = self.preprocessor_.transform(X)

        # 3) find positive / negative indices
        pos_idx = np.flatnonzero(y.values == 1)
        neg_idx = np.flatnonzero(y.values == 0)

        P = len(pos_idx)
        N = len(neg_idx)
        n_models = N // P

        # 4) shuffle & chunk negatives
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(neg_idx)
        neg_chunks = np.array_split(neg_idx, n_models)

        # 5) for each chunk build & fit one classifier
        self.estimators_ = []
        for chunk in neg_chunks:
            idx = np.concatenate([pos_idx, chunk])
            Xb = X_all[idx]
            yb = y.values[idx]

            est = clone(self.base_estimator)
            est.fit(Xb, yb)
            self.estimators_.append(est)

        return self

    def predict_proba(self, X):
        # apply the *same* preprocessor
        Xp = self.preprocessor_.transform(X)
        # collect all the per‐estimator proba[:,1]
        probs = np.stack([m.predict_proba(Xp)[:, 1] for m in self.estimators_], axis=1)
        # average them
        mean_pos = probs.mean(axis=1)
        return np.vstack([1 - mean_pos, mean_pos]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:15:38 2024

@author: Elena
"""
# Necessary bibliothek
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from category_encoders import TargetEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

# Load data
train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

train_dtypes = train_df.dtypes
test_dtypes = test_df.dtypes

# %%

# Selection of numerical and categorical features
numerical_features = train_dtypes[train_dtypes == 'float64'].index.tolist() + train_dtypes[train_dtypes == 'int64'].index.tolist()
categorical_features = train_dtypes[train_dtypes == 'object'].index.tolist()

# Selection of the target 'y' from the train dataset
target_column = 'y'
train_df_without_target = train_df.drop(columns=[target_column])

# Remove columns with NaN more 35%
def remove_columns_with_high_nan(df, threshold=0.35):
    threshold_count = int(threshold * len(df))
    df_cleaned = df.dropna(thresh=len(df) - threshold_count, axis=1)
    return df_cleaned

# Cleaned dates in train and test data
train_df_cleaned = remove_columns_with_high_nan(train_df_without_target)
test_df_cleaned = remove_columns_with_high_nan(test_df)

# Update after deleting columns with NaN
numerical_features_cleaned = [col for col in numerical_features if col in train_df_cleaned.columns]

# %%
# Continue data preparation
# Based on the fact that the dataset has columns with a large number of NaN values, it was decided to impute numerical features using KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)
train_df_cleaned[numerical_features_cleaned] = knn_imputer.fit_transform(train_df_cleaned[numerical_features_cleaned])
test_df_cleaned[numerical_features_cleaned] = knn_imputer.transform(test_df_cleaned[numerical_features_cleaned])

# Imputation for categorical features (replacing missing values with the mode)
categorical_imputer = SimpleImputer(strategy='most_frequent')
for col in categorical_features:
    if col in train_df_cleaned.columns:
        train_df_cleaned[[col]] = categorical_imputer.fit_transform(train_df_cleaned[[col]])
    if col in test_df_cleaned.columns:
        test_df_cleaned[[col]] = categorical_imputer.transform(test_df_cleaned[[col]])

# Checking that there are no more missing values
missing_values_train_after = train_df_cleaned.isnull().sum().sum()
missing_values_test_after = test_df_cleaned.isnull().sum().sum()

# Returning the target variable "y" to the cleaned training dataset
train_df_cleaned[target_column] = train_df[target_column]

# %%

# Splitting into training and validation sets
X = train_df_cleaned.drop(columns=[target_column])
y = train_df_cleaned[target_column]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# A function to combine rare categories will help avoid overfitting
def simplify_categories(df, column, threshold=0.01):
    # Розрахувати частоту категорій
    category_frequencies = df[column].value_counts(normalize=True)
    # Об'єднати категорії, які мають частоту меншу за поріг
    rare_categories = category_frequencies[category_frequencies < threshold].index
    df[column] = df[column].apply(lambda x: 'Other' if x in rare_categories else x)
 
# Apply simplification to all categorical features with a large number of categories
for col in categorical_features:
    if col in train_df_cleaned.columns and col in test_df_cleaned.columns:
        unique_values = train_df_cleaned[col].nunique()
        if unique_values > 50:  
            simplify_categories(train_df_cleaned, col)
            simplify_categories(test_df_cleaned, col)
            
# Encoding categorical features using TargetEncoder
target_encoder = TargetEncoder()
X_train_encoded = target_encoder.fit_transform(X_train, y_train)
X_val_encoded = target_encoder.transform(X_val)

# Normalizing numerical features using PowerTransformer
power_transformer = PowerTransformer()
X_train_encoded[numerical_features_cleaned] = power_transformer.fit_transform(X_train_encoded[numerical_features_cleaned])
X_val_encoded[numerical_features_cleaned] = power_transformer.transform(X_val_encoded[numerical_features_cleaned])

# Converting back to a DataFrame with the corresponding column names
X_train_final = pd.DataFrame(X_train_encoded, columns=X_train.columns)
X_val_final = pd.DataFrame(X_val_encoded, columns=X_val.columns)

# Normalizing numerical features using StandardScaler
scaler = StandardScaler()
X_train_scaled = X_train_final.copy()
X_val_scaled = X_val_final.copy()

X_train_scaled[numerical_features_cleaned] = scaler.fit_transform(X_train_scaled[numerical_features_cleaned])
X_val_scaled[numerical_features_cleaned] = scaler.transform(X_val_scaled[numerical_features_cleaned])

# Balancing data using SMOTE or SMOTEENN worsened the subsequent model training (possibly due to the synthetic nature of the data). Therefore, it was decided 
# to opt for Undersampling 
# Class balancing using RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train_scaled, y_train)

# %%

# Training models using algorithms that are robust to imbalanced data
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)
y_val_pred_rf = rf_model.predict(X_val_scaled)
rf_accuracy = accuracy_score(y_val, y_val_pred_rf)
print(f"Random Forest accuracy: {rf_accuracy}")

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_balanced, y_train_balanced)
y_val_pred_gb = gb_model.predict(X_val_scaled)
gb_accuracy = accuracy_score(y_val, y_val_pred_gb)
print(f"Gradient Boosting accuracy: {gb_accuracy}")


# %%
# Creating a model for training using Gradient Boosting, as it showed the best result
# Pipeline with Undersampling
pipeline = ImbPipeline([
    ('undersampler', RandomUnderSampler(random_state=42)),
    ('scaler', StandardScaler()),  
    ('classifier', StackingClassifier(estimators=[
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                          max_depth=5, random_state=42)),
            ], final_estimator=LogisticRegression()))
])

# Creating a StratifiedKFold object
stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Performing cross-validation using StratifiedKFold
cv_scores = cross_val_score(pipeline, X_train_balanced, y_train_balanced, cv=stratified_kf, scoring='accuracy', n_jobs=-1)

print(f"Stratified K-Fold Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# The cross-validation result with an accuracy of 0.9736 ± 0.0069 is very high and indicates that the model demonstrates stable performance across different subsets of the data.
# The low standard deviation (0.0069) suggests that the model performs consistently without significant fluctuations in accuracy between different folds, which is a positive sign.
# %%

# Combining all data for model training and retraining the final model on all available data
X_combined = pd.concat([X_train_balanced, X_val_scaled], axis=0)
y_combined = pd.concat([y_train_balanced, y_val], axis=0)

pipeline.fit(X_combined, y_combined)

# %%

# Checking that the test set has the same columns as the training set
missing_cols = set(X_train_balanced.columns) - set(test_df_cleaned.columns)
for col in missing_cols:
    # If a column is missing, fill it with the mean value (for numerical features) or the mode (for categorical features)
    if col in numerical_features_cleaned:
        test_df_cleaned[col] = X_train_balanced[col].mean()
    else:
        test_df_cleaned[col] = X_train_balanced[col].mode()[0]

# Reordering the columns in the test set to match the training set
test_df_cleaned = test_df_cleaned[X_train_balanced.columns]

# Checking for new or unknown categories that were not in the training set
for col in categorical_features:
    if col in test_df_cleaned.columns:
        unknown_categories = set(test_df_cleaned[col].unique()) - set(X_train[col].unique())
        if unknown_categories:
            # Replacing unknown categories with a special value, e.g., 'Unknown'
            test_df_cleaned[col] = test_df_cleaned[col].apply(lambda x: 'Unknown' if x in unknown_categories else x)

# Checking for new or unknown categories that were not in the training set
test_df_encoded = target_encoder.transform(test_df_cleaned)

# Normalizing numerical features using PowerTransformer (same as on the training data)
test_df_encoded[numerical_features_cleaned] = power_transformer.transform(test_df_encoded[numerical_features_cleaned])

# Scaling numerical features using StandardScaler
test_df_scaled = test_df_encoded.copy()
test_df_scaled[numerical_features_cleaned] = scaler.transform(test_df_scaled[numerical_features_cleaned])

# Generating predictions for the test set
y_test_pred = pipeline.predict(test_df_scaled)

# %%
# Loading the sample file and saving the file in the specified directory
sample_submission = pd.read_csv(sample_submission_path)


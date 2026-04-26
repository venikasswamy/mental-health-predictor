import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# LOAD DATA
# -------------------------
data = pd.read_csv("../data/dataset.csv")

# -------------------------
# FIX COLUMN NAMES
# -------------------------
data.columns = data.columns.str.strip().str.lower()

# -------------------------
# BASIC CLEANING
# -------------------------
data = data.dropna(how='all')
data = data.drop_duplicates()

print("✅ Basic cleaning done")

# -------------------------
# TARGET COLUMN
# -------------------------
target = "depression"

if target not in data.columns:
    raise Exception(f"❌ Target column '{target}' not found.\nColumns: {list(data.columns)}")

X = data.drop(target, axis=1)
y = data[target]

# Convert target if needed (Yes/No → 1/0)
if y.dtype == 'object':
    y = y.str.lower().map({'yes': 1, 'no': 0})

# -------------------------
# DROP LOW-IMPORTANCE COLUMNS
# -------------------------
X = X.drop([
    "university",
    "degree_major",
    "degree_level",
    "academic_year"
], axis=1, errors='ignore')

# -------------------------
# FEATURE TYPES
# -------------------------
categorical = X.select_dtypes(include=['object', 'string']).columns
numerical = X.select_dtypes(exclude=['object', 'string']).columns

# -------------------------
# PREPROCESSING PIPELINE (FIXED)
# -------------------------
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical)
])

# -------------------------
# MODELS
# -------------------------
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ),

    "LogisticRegression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ),

    "SVM": SVC(
        probability=True,
        class_weight="balanced"
    ),

    "HGB": HistGradientBoostingClassifier()
}

# Create models folder
os.makedirs("../models", exist_ok=True)

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# TRAIN & SAVE
# -------------------------
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    with open(f"../models/{name}.pkl", "wb") as f:
        pickle.dump(pipeline, f)

print("\n✅ Training complete. Models saved in /models")
"""
CBE AI-Based Credit Scoring System - Role Based Dashboard
Run:
    streamlit run cbe_credit_scoring_role_based.py

Files expected in same folder:
    loan_data.csv  OR  loan_data.csv
    cbe_logo.png   optional

Default demo users:
    admin / admin123      -> Admin
    tigist / tigist123    -> Loan Officer
    surafel / surafel123  -> Authorizer / Manager
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import warnings
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Tuple

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
from sklearn.impute import SimpleImputer
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PAGE SETTINGS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CBE Credit Scoring System", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")

TARGET = "loan_status"
ID_COL = "customer_id"
DEFAULT_DATA_FILES = ["loan_data.csv", "loan_data.csv"]
LOGO_FILES = ["cbe_logo.png", "CBE_logo.png", "cbe.png", "logo.png"]
USER_DB_FILE = "cbe_users.json"

FIELD_DESCRIPTIONS: Dict[str, str] = {
    "customer_id": "Unique identifier for each customer / loan application",
    "borrower_type": "Borrower category: Individual or Corporate",
    "borrower_age_or_company_age": "Age of individual borrower or age of company in years",
    "gender_or_company_entity_type": "Gender for individuals or legal/entity type for companies",
    "education_or_company_size": "Education level for individuals or size/status level for companies",
    "income": "Annual borrower or business income",
    "employment_experience_or_years_in_business": "Years of employment experience or years in business",
    "home_ownership_or_business_premises": "Home ownership or business premises status",
    "loan_amnt": "Requested or approved loan amount",
    "loan_intent": "Purpose / intent of the loan",
    "loan_int_rate": "Interest rate charged on the loan",
    "loan_percent_income": "Loan amount as percentage of income",
    "credit_history_length_years": "Length of credit history in years",
    "credit_score": "Borrower credit score",
    "previous_loan_defaults_on_file": "Whether borrower has previous default / NPL on file",
    "loan_status": "Target: 0 = Good / Performing, 1 = Bad / NPL",
}

ALL_PAGES = [
    "🏦 Loan Officer Dashboard",
    "📦 Bulk Credit Scoring",
    "🧮 Credit Scoring Calculator",
    "📄 Loan Officer Report",
    "👔 Executive Dashboard",
    "📑 Executive Report",
    "📋 Dataset Overview",
    "🔍 Exploratory Data Analysis",
    "⚙️ Model Training & Evaluation",
    "🎯 Best Model Results",
    "👥 Admin User & GUI Management",
]

DEFAULT_ROLE_PAGES = {
    "Admin": [
        "📋 Dataset Overview",
        "🔍 Exploratory Data Analysis",
        "⚙️ Model Training & Evaluation",
        "🎯 Best Model Results",
        "👥 Admin User & GUI Management",
    ],
    "Loan Officer": [
        "🏦 Loan Officer Dashboard",
        "🧮 Credit Scoring Calculator",
        "📦 Bulk Credit Scoring",
        "📄 Loan Officer Report",
    ],
    "Authorizer / Manager": [
        "👔 Executive Dashboard",
        "🧮 Credit Scoring Calculator",
        "📦 Bulk Credit Scoring",
        "📑 Executive Report",
    ],
}

# -----------------------------------------------------------------------------
# CSS DESIGN
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #210047 0%, #3B006D 45%, #6A0DAD 100%); color: #ffffff; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1b003b 0%, #340067 55%, #4b0082 100%); border-right: 2px solid rgba(255, 215, 0, 0.55); }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    .main .block-container { max-width: 100% !important; padding: 1.1rem 1.6rem 2rem 1.6rem; }
    .cbe-hero { background: rgba(255,255,255,0.95); border: 2px solid rgba(255,215,0,0.85); border-radius: 24px; padding: 18px 20px 14px 20px; margin-bottom: 16px; box-shadow: 0 8px 28px rgba(0,0,0,0.28); text-align: center; }
    .cbe-title { color: #3B006D !important; font-size: 42px; font-weight: 900; margin: 5px 0 0 0; letter-spacing: .3px; }
    .cbe-subtitle { color: #B8860B !important; font-size: 24px; font-weight: 800; margin-top: -2px; }
    .section-card { background: rgba(255,255,255,0.96); color: #1f1f1f; border-radius: 20px; border: 1px solid rgba(255,215,0,0.45); padding: 20px; margin: 12px 0 18px 0; box-shadow: 0 8px 24px rgba(0,0,0,0.22); }
    .section-card h1, .section-card h2, .section-card h3, .section-card p, .section-card label { color: #1f1f1f !important; }
    div[data-testid="stMetric"] { background: linear-gradient(135deg, #fffaf0 0%, #ffffff 100%); border: 1px solid rgba(255,215,0,0.75); border-radius: 18px; padding: 14px; box-shadow: 0 5px 16px rgba(0,0,0,0.12); }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] div { color: #2D0060 !important; font-weight: 800; }
    .good-box { background: linear-gradient(135deg, #0d6b2b 0%, #28a745 100%); padding: 24px; border-radius: 18px; border: 2px solid #FFD700; color: #ffffff; margin: 12px 0; }
    .bad-box { background: linear-gradient(135deg, #8f1020 0%, #dc3545 100%); padding: 24px; border-radius: 18px; border: 2px solid #FFD700; color: #ffffff; margin: 12px 0; }
    .info-box { background: linear-gradient(135deg, #075c68 0%, #17a2b8 100%); padding: 24px; border-radius: 18px; border: 2px solid #FFD700; color: #ffffff; margin: 12px 0; }
    .gold-note { background: linear-gradient(135deg, #FFD700 0%, #FFC107 100%); color: #2D0060; padding: 12px 16px; border-radius: 14px; font-weight: 800; margin: 10px 0; }
    .stButton > button, .stDownloadButton > button { background: linear-gradient(135deg, #FFD700 0%, #FFC107 100%) !important; color: #2D0060 !important; font-weight: 900 !important; border-radius: 12px !important; border: 1px solid #fff2a8 !important; padding: 0.7rem 1.2rem !important; }
    .stTabs [data-baseweb="tab"] { background: rgba(255,255,255,0.92); border-radius: 12px; color: #2D0060; font-weight: 800; border: 1px solid rgba(255,215,0,.45); }
    h1, h2, h3 { color: #FFD700 !important; font-weight: 900 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# AUTHENTICATION AND ROLE MANAGEMENT
# -----------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def default_users() -> Dict:
    return {
        "users": {
            "admin": {"password_hash": hash_password("admin123"), "role": "Admin", "full_name": "System Administrator", "active": True},
            "tigist": {"password_hash": hash_password("tigist123"), "role": "Loan Officer", "full_name": "Tigist Mola", "active": True},
            "surafel": {"password_hash": hash_password("surafel123"), "role": "Authorizer / Manager", "full_name": "Surafel Aman", "active": True},
        },
        "role_pages": DEFAULT_ROLE_PAGES,
    }


def load_user_db() -> Dict:
    """Load user database safely and always keep default demo users available."""
    default_db = default_users()

    if not os.path.exists(USER_DB_FILE):
        save_user_db(default_db)
        return default_db

    try:
        with open(USER_DB_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)
    except Exception:
        db = default_db

    db.setdefault("users", {})
    db.setdefault("role_pages", DEFAULT_ROLE_PAGES)

    # Repair default demo users if the saved JSON was edited, missing, or inactive.
    for username, info in default_db["users"].items():
        if username not in db["users"]:
            db["users"][username] = info
        else:
            db["users"][username]["password_hash"] = info["password_hash"]
            db["users"][username]["role"] = info["role"]
            db["users"][username]["full_name"] = info["full_name"]
            db["users"][username]["active"] = True

    for role, pages in DEFAULT_ROLE_PAGES.items():
        existing_pages = db["role_pages"].setdefault(role, [])
        for page in pages:
            if page not in existing_pages:
                existing_pages.append(page)

        if role == "Loan Officer":
            if "🏦 Dashboard" in existing_pages:
                existing_pages.remove("🏦 Dashboard")
            if "📄 Report" in existing_pages:
                existing_pages.remove("📄 Report")
            if "🏦 Loan Officer Dashboard" not in existing_pages:
                existing_pages.insert(0, "🏦 Loan Officer Dashboard")
            if "📄 Loan Officer Report" not in existing_pages:
                existing_pages.append("📄 Loan Officer Report")

    save_user_db(db)
    return db


def save_user_db(db: Dict):
    with open(USER_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def authenticate(username: str, password: str) -> Tuple[bool, str, str]:
    db = load_user_db()
    username = username.strip().lower()
    password = password.strip()
    user = db["users"].get(username)
    if not user or not user.get("active", True):
        return False, "", ""
    if user.get("password_hash") == hash_password(password):
        return True, user.get("role", "Loan Officer"), user.get("full_name", username)
    return False, "", ""


def render_login():
    logo_html = get_logo_html(110)
    st.markdown(
        f"""
        <div class="cbe-hero" style="max-width:520px; margin:70px auto 15px auto;">
            {logo_html}
            <div class="cbe-title" style="font-size:30px;">Commercial Bank of Ethiopia</div>
            <div class="cbe-subtitle" style="font-size:18px;">AI-Based Credit Scoring System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("🔐 LOGIN", use_container_width=True)
            if submitted:
                ok, role, full_name = authenticate(username.strip(), password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()
                    st.session_state.role = role
                    st.session_state.full_name = full_name
                    st.rerun()
                else:
                    st.error("Invalid username or password, or the user is inactive.")
        #st.caption("Demo credentials: admin/admin123, tigist/tigist123, surafel/surafel123")
       # if st.button("🔄 Reset Demo Users", use_container_width=True):
        #    save_user_db(default_users())
         #   st.success("Demo users reset. Now log in using admin/admin123, tigist/tigist123, or surafel/surafel123.")
          #  st.rerun()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------------------------------------------------------------
# GENERAL HELPERS
# -----------------------------------------------------------------------------
def find_first_existing(paths: List[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def get_logo_html(width: int = 105) -> str:
    logo_path = find_first_existing(LOGO_FILES)
    if logo_path:
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{encoded}" width="{width}" style="margin-bottom: 4px;" />'
    return '<div style="font-size:64px; line-height:70px;">🏦</div>'


def render_logo_header(page_title: str):
    st.markdown(
        f"""
        <div class="cbe-hero">
            {get_logo_html(105)}
            <div class="cbe-title">Commercial Bank of Ethiopia</div>
            <div class="cbe-subtitle">✨ AI-Based Credit Scoring and Loan Portfolio Dashboard ✨</div>
            <div style="color:#3B006D; font-weight:800; margin-top:8px;">{page_title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def open_card():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)


def close_card():
    st.markdown('</div>', unsafe_allow_html=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    for file_name in DEFAULT_DATA_FILES:
        if os.path.exists(file_name):
            return normalize_columns(pd.read_csv(file_name))
    raise FileNotFoundError("No CSV found. Upload the file from the sidebar, or place loan_data.csv beside this script.")


def safe_quantile_segment(series: pd.Series, labels: list[str], fallback_label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(fallback_label, index=series.index, dtype="object")
    valid = numeric.dropna()
    if len(valid) < 2 or valid.nunique() < 2:
        out.loc[valid.index] = fallback_label
        return out.fillna("Unknown")
    try:
        ranked = numeric.rank(method="first")
        q = min(len(labels), int(valid.nunique()), len(valid))
        if q < 2:
            return out.fillna("Unknown")
        chosen_labels = labels[:q]
        return pd.qcut(ranked, q=q, labels=chosen_labels, duplicates="drop").astype("object").fillna("Unknown")
    except Exception:
        return out.fillna("Unknown")


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
        df["credit_grade"] = np.where(df[TARGET] == 1, "Bad / NPL", "Good / Performing")
    if "borrower_age_or_company_age" in df.columns:
        df["age_or_company_age_band"] = pd.cut(
            pd.to_numeric(df["borrower_age_or_company_age"], errors="coerce"),
            bins=[0, 25, 35, 45, 60, 10000],
            labels=["<=25", "26-35", "36-45", "46-60", "60+"],
            include_lowest=True,
        ).astype("object").fillna("Unknown")
    if "income" in df.columns:
        df["income_level"] = safe_quantile_segment(df["income"], ["Very Low", "Low", "Medium", "High", "Very High"], "Single Customer")
    if "loan_amnt" in df.columns:
        df["loan_amount_level"] = safe_quantile_segment(df["loan_amnt"], ["Very Small", "Small", "Medium", "Large", "Very Large"], "Single Loan")
    if "application_stage" not in df.columns:
        df["application_stage"] = "Existing portfolio - no workflow stage in CSV"
    if "granted_date" not in df.columns:
        df["granted_date"] = pd.NaT
    if "request_date" not in df.columns:
        df["request_date"] = pd.NaT
    return df


def format_money(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.0f}"


def group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns or TARGET not in df.columns:
        return pd.DataFrame()
    amount_col = "loan_amnt" if "loan_amnt" in df.columns else TARGET
    out = (
        df.groupby(group_col, dropna=False)
        .agg(
            total_loans=(TARGET, "size"),
            good_loans=(TARGET, lambda s: int((s == 0).sum())),
            npl_loans=(TARGET, lambda s: int((s == 1).sum())),
            npl_rate_pct=(TARGET, lambda s: round(float(s.mean() * 100), 2)),
            total_amount=(amount_col, "sum"),
            avg_amount=(amount_col, "mean"),
        )
        .reset_index()
        .sort_values("total_loans", ascending=False)
    )
    out["total_amount"] = out["total_amount"].map(format_money)
    out["avg_amount"] = out["avg_amount"].map(format_money)
    return out


def plot_bar(data: pd.DataFrame, x: str, y: str, title: str, ylabel: str = ""):
    if data.empty:
        st.info("No data available for this chart.")
        return
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    ax.bar(data[x].astype(str), data[y])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel or y, fontsize=9)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def plot_pie(labels, values, title: str):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    ax.set_title(title, fontsize=11, fontweight="bold")
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    model_df = df.copy()
    y = model_df[TARGET].astype(int)
    drop_cols = [TARGET, "credit_grade", "application_stage", "granted_date", "request_date"]
    if ID_COL in model_df.columns:
        drop_cols.append(ID_COL)
    X = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, y, numeric_cols, categorical_cols


@st.cache_resource(show_spinner=False)
def train_models_cached(df: pd.DataFrame):
    X, y, numeric_cols, categorical_cols = prepare_features(df)
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)])
    models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Tree": ExtraTreeClassifier(
        max_depth=10,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42
    ),
}
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(random_state=42, n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred
        results[name] = {
            "pipeline": pipe,
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_proba)),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    best_name = max(results, key=lambda k: results[k]["f1"])
    return results, best_name, X.columns.tolist()


def feature_importance_table(best_pipeline: Pipeline, feature_columns: List[str]) -> pd.DataFrame:
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array(feature_columns)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()
    return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(20)


def predict_customer(best_pipeline: Pipeline, raw_df: pd.DataFrame, customer_id: str):
    full_segmented = add_segments(raw_df.copy())
    match = full_segmented[full_segmented[ID_COL].astype(str).str.upper() == customer_id.upper()]
    if match.empty:
        return None
    X, _, _, _ = prepare_features(full_segmented.copy())
    row_X = match.reindex(columns=X.columns)
    pred = int(best_pipeline.predict(row_X)[0])
    proba = float(best_pipeline.predict_proba(row_X)[0, 1]) if hasattr(best_pipeline, "predict_proba") else float(pred)
    return match.iloc[0], pred, proba


def score_dataframe(best_pipeline: Pipeline, train_df: pd.DataFrame, scoring_df: pd.DataFrame) -> pd.DataFrame:
    """Score a bulk CSV.

    Works in two common cases:
    1) The uploaded CSV contains full borrower/application columns.
    2) The uploaded CSV contains only customer_id values; the app fills details from the approved portfolio.

    Rows with customer IDs not found and no scoring fields are returned as No Data / New Customer
    instead of crashing or producing misleading predictions.
    """
    train_segmented = add_segments(train_df.copy())
    X_template, _, _, _ = prepare_features(train_segmented)
    model_cols = X_template.columns.tolist()

    upload = normalize_columns(scoring_df.copy())
    scoring_segmented = add_segments(upload.copy())

    # If customer_id is supplied, use uploaded values first and fill missing values from historical data.
    no_data_mask = pd.Series(False, index=scoring_segmented.index)
    if ID_COL in scoring_segmented.columns and ID_COL in train_segmented.columns:
        hist = train_segmented.drop_duplicates(ID_COL).set_index(train_segmented[ID_COL].astype(str).str.upper())
        ids = scoring_segmented[ID_COL].astype(str).str.upper()
        hist_rows = hist.reindex(ids).reset_index(drop=True)
        scoring_segmented = scoring_segmented.reset_index(drop=True).combine_first(hist_rows)
        no_data_mask = ids.map(lambda x: x not in hist.index)

    X_score = scoring_segmented.reindex(columns=model_cols)
    enough_data = ~X_score.isna().all(axis=1)
    predict_mask = enough_data & ~no_data_mask

    result = upload.copy().reset_index(drop=True)
    result["predicted_loan_status"] = pd.NA
    result["predicted_credit_grade"] = "No Data / New Customer"
    result["probability_bad_npl"] = pd.NA
    result["recommended_action"] = "Proceed with alternative credit assessment, request supporting documents, and consider collateral/guarantor options"

    if predict_mask.any():
        pred = best_pipeline.predict(X_score.loc[predict_mask])
        proba = best_pipeline.predict_proba(X_score.loc[predict_mask])[:, 1] if hasattr(best_pipeline, "predict_proba") else pred
        result.loc[predict_mask, "predicted_loan_status"] = pred.astype(int)
        result.loc[predict_mask, "predicted_credit_grade"] = np.where(pred == 1, "Bad / NPL", "Good / Performing")
        result.loc[predict_mask, "probability_bad_npl"] = np.round(proba, 4)
        result.loc[predict_mask, "recommended_action"] = np.where(
            pred == 1,
            "Reject or require enhanced collateral/guarantor and close monitoring",
            "Proceed with standard loan approval process",
        )

    return result


def make_excel_download(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, data in sheets.items():
            data.to_excel(writer, index=False, sheet_name=name[:31])
    return output.getvalue()


def basic_report_tables(df: pd.DataFrame, title: str) -> Dict[str, pd.DataFrame]:
    summary = pd.DataFrame([
        {"Metric": "Report", "Value": title},
        {"Metric": "Generated At", "Value": datetime.now().strftime("%Y-%m-%d %H:%M")},
        {"Metric": "Total Loans", "Value": len(df)},
        {"Metric": "Good / Performing", "Value": int((df[TARGET] == 0).sum()) if TARGET in df.columns else "-"},
        {"Metric": "Bad / NPL", "Value": int((df[TARGET] == 1).sum()) if TARGET in df.columns else "-"},
        {"Metric": "NPL Rate", "Value": f"{float(df[TARGET].mean() * 100):.2f}%" if TARGET in df.columns else "-"},
    ])
    tables = {"Summary": summary}
    for col in ["borrower_type", "loan_intent", "income_level", "loan_amount_level", "education_or_company_size", "gender_or_company_entity_type"]:
        if col in df.columns and TARGET in df.columns:
            tables[col[:31]] = group_summary(df, col)
    return tables

# -----------------------------------------------------------------------------
# LOGIN
# -----------------------------------------------------------------------------
if not st.session_state.logged_in:
    render_login()
    st.stop()

user_db = load_user_db()
role = st.session_state.get("role", "Loan Officer")
username = st.session_state.get("username", "")
full_name = st.session_state.get("full_name", username)
allowed_pages = user_db.get("role_pages", {}).get(role, DEFAULT_ROLE_PAGES.get(role, []))

st.sidebar.success(f"✅ {full_name}\n\nRole: {role}")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.full_name = None
    st.rerun()

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
st.sidebar.markdown("# 📊 Navigation")
st.sidebar.markdown("Pages are controlled by your role privileges.")

if role == "Authorizer / Manager":
    st.sidebar.info("Manager access includes Executive Dashboard, Credit Scoring Calculator, Bulk Credit Scoring, and Executive Report.")
elif role == "Loan Officer":
    st.sidebar.info("Loan Officer access includes Dashboard, Credit Scoring Calculator, Bulk Credit Scoring, and Report Generation.")
elif role == "Admin":
    st.sidebar.info("Admin access includes User Management, Dataset Overview, EDA, Model Training, and Best Model Results.")

section = st.sidebar.radio("Go to:", allowed_pages)

# Admin can upload the main dataset. Other users use the approved/default dataset.
if role == "Admin":
    uploaded_main = st.sidebar.file_uploader("Upload approved loan CSV Database", type=["csv"], key="admin_main_csv")
else:
    uploaded_main = None

try:
    if uploaded_main is not None:
        df_raw = normalize_columns(pd.read_csv(uploaded_main))
    else:
        df_raw = load_default_data()
    df = add_segments(df_raw)
except Exception as exc:
    render_logo_header("Data loading")
    st.error(str(exc))
    st.stop()

if TARGET not in df.columns:
    render_logo_header("Missing target column")
    st.error(f"The dataset must contain `{TARGET}` where 0 = Good and 1 = Bad/NPL.")
    st.stop()

# Dashboard filters are available for Admin and Authorizer/Manager only.
# Loan officers see their ordinary operational dashboard without sidebar filters.
filtered_df = df.copy()
#if role != "Loan Officer":
 #   st.sidebar.markdown("---")
  #  st.sidebar.markdown("## 🔎 Dashboard Filters")
   # for filter_col in ["borrower_type", "loan_intent", "gender_or_company_entity_type", "education_or_company_size"]:
    #    if filter_col in filtered_df.columns:
     #       options = sorted(filtered_df[filter_col].dropna().astype(str).unique().tolist())
      #      selected = st.sidebar.multiselect(filter_col.replace("_", " ").title(), options, default=options)
       #     if selected:
        #        filtered_df = filtered_df[filtered_df[filter_col].astype(str).isin(selected)]

# Ensure section is always defined before page rendering.
if "section" not in locals():
    section = allowed_pages[0] if allowed_pages else "👔 Executive Dashboard"

# -----------------------------------------------------------------------------
# PAGE: ADMIN USER AND GUI MANAGEMENT
# -----------------------------------------------------------------------------
if section == "👥 Admin User & GUI Management":
    render_logo_header("Admin User and GUI Privilege Management")
    open_card()
    st.header("👥 Admin Panel")
    st.markdown('<div class="gold-note">Admin can create users, remove/deactivate users, and control which GUI pages each role can access.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Create User", "Manage Users", "GUI Page Privileges"])
    with tab1:
        with st.form("create_user_form"):
            new_username = st.text_input("New username")
            new_full_name = st.text_input("Full name")
            new_role = st.selectbox("Role", list(DEFAULT_ROLE_PAGES.keys()))
            new_password = st.text_input("Temporary password", type="password")
            create = st.form_submit_button("Create User")
            if create:
                db = load_user_db()
                if not new_username or not new_password:
                    st.error("Username and password are required.")
                elif new_username in db["users"]:
                    st.error("Username already exists.")
                else:
                    db["users"][new_username] = {"password_hash": hash_password(new_password), "role": new_role, "full_name": new_full_name or new_username, "active": True}
                    save_user_db(db)
                    st.success(f"User `{new_username}` created successfully.")

    with tab2:
        db = load_user_db()
        users_df = pd.DataFrame([
            {"username": u, "full_name": v.get("full_name"), "role": v.get("role"), "active": v.get("active", True)}
            for u, v in db["users"].items()
        ])
        st.dataframe(users_df, use_container_width=True)
        selected_user = st.selectbox("Select user to update", users_df["username"].tolist())
        if selected_user:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Deactivate / Remove Access") and selected_user != "admin":
                    db["users"][selected_user]["active"] = False
                    save_user_db(db)
                    st.success("User deactivated.")
                    st.rerun()
            with c2:
                if st.button("Reactivate"):
                    db["users"][selected_user]["active"] = True
                    save_user_db(db)
                    st.success("User reactivated.")
                    st.rerun()
            with c3:
                if st.button("Delete User") and selected_user != "admin":
                    db["users"].pop(selected_user, None)
                    save_user_db(db)
                    st.success("User deleted.")
                    st.rerun()

    with tab3:
        db = load_user_db()
        selected_role = st.selectbox("Select role", list(DEFAULT_ROLE_PAGES.keys()))
        current_pages = db.get("role_pages", {}).get(selected_role, DEFAULT_ROLE_PAGES[selected_role])
        new_pages = st.multiselect("GUI pages allowed for this role", ALL_PAGES, default=current_pages)
        if st.button("Save GUI Privileges"):
            db.setdefault("role_pages", {})[selected_role] = new_pages
            save_user_db(db)
            st.success("GUI privileges updated. Affected users should log out and log in again.")
    close_card()

# -----------------------------------------------------------------------------
# PAGE: LOAN OFFICER DASHBOARD
# -----------------------------------------------------------------------------
elif section == "🏦 Loan Officer Dashboard":
    render_logo_header("Loan Officer Dashboard")
    open_card()
    st.header("🏦 Operational Loan Officer Dashboard")
    total = len(filtered_df)
    good = int((filtered_df[TARGET] == 0).sum())
    bad = int((filtered_df[TARGET] == 1).sum())
    npl_rate = bad / total * 100 if total else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assigned / Filtered Loans", f"{total:,}")
    c2.metric("Good / Performing", f"{good:,}")
    c3.metric("Bad / NPL", f"{bad:,}")
    c4.metric("NPL Rate", f"{npl_rate:.2f}%")
    tabs = st.tabs(["Daily Work View", "NPL View", "Segment Table"])
    with tabs[0]:
        if "borrower_type" in filtered_df.columns:
            plot_bar(group_summary(filtered_df, "borrower_type"), "borrower_type", "total_loans", "Loans by Borrower Type", "Count")
        st.dataframe(filtered_df.head(50), use_container_width=True)
    with tabs[1]:
        for col in ["loan_intent", "income_level", "loan_amount_level", "previous_loan_defaults_on_file"]:
            if col in filtered_df.columns:
                plot_bar(group_summary(filtered_df, col), col, "npl_rate_pct", f"NPL Rate by {col.replace('_',' ').title()}", "NPL Rate (%)")
    with tabs[2]:
        cols = [c for c in ["borrower_type", "loan_intent", "income_level", "loan_amount_level", "education_or_company_size"] if c in filtered_df.columns]
        selected_segment = st.selectbox("Select segment", cols)
        st.dataframe(group_summary(filtered_df, selected_segment), use_container_width=True)
    close_card()

# -----------------------------------------------------------------------------
# PAGE: BULK CREDIT SCORING
# -----------------------------------------------------------------------------
elif section == "📦 Bulk Credit Scoring":
    render_logo_header("Bulk Credit Scoring Analysis")
    open_card()
    st.header("📦 Upload Bulk Applications for AI Credit Scoring")
    st.markdown('<div class="gold-note">Upload a CSV, preview it, then click Run Bulk Credit Scoring. The file can contain full application fields or only a customer_id column.</div>', unsafe_allow_html=True)
    results, best_name, feature_cols = train_models_cached(df)
    best_pipeline = results[best_name]["pipeline"]

    with st.expander("CSV format help", expanded=False):
        st.write("Best option: upload the same columns as the loan dataset, without needing loan_status. For existing customers, a CSV with only customer_id also works.")
        if ID_COL in df_raw.columns:
            sample_template = pd.DataFrame({ID_COL: df_raw[ID_COL].astype(str).head(5).tolist()})
            st.dataframe(sample_template, use_container_width=True)
            st.download_button("⬇️ Download simple customer_id template", sample_template.to_csv(index=False).encode("utf-8"), "bulk_customer_id_template.csv", "text/csv")

    bulk_file = st.file_uploader("Upload bulk customer/application CSV", type=["csv"], key="bulk_upload")

    if bulk_file is not None:
        try:
            bulk_df = normalize_columns(pd.read_csv(bulk_file))
            st.subheader("Uploaded CSV Preview")
            st.dataframe(bulk_df.head(20), use_container_width=True)

            if st.button("🚀 Run Bulk Credit Scoring", use_container_width=True):
                if bulk_df.empty:
                    st.error("The uploaded CSV is empty.")
                else:
                    with st.spinner("Running bulk credit scoring..."):
                        scored = score_dataframe(best_pipeline, df_raw, bulk_df)
                        st.session_state["bulk_scored_results"] = scored
                        st.session_state["bulk_best_model"] = best_name
                    st.success(f"Bulk scoring completed using {best_name}.")

            scored = st.session_state.get("bulk_scored_results")
            if scored is not None:
                st.subheader("Bulk Scoring Results")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Applications Scored", f"{len(scored):,}")
                c2.metric("Predicted Good", f"{int((scored['predicted_loan_status']==0).sum()):,}")
                c3.metric("Predicted Bad/NPL", f"{int((scored['predicted_loan_status']==1).sum()):,}")
                c4.metric("No Data / New", f"{int(scored['predicted_loan_status'].isna().sum()):,}")
                plot_pie(scored["predicted_credit_grade"].value_counts().index, scored["predicted_credit_grade"].value_counts().values, "Bulk Scoring Result Mix")
                st.dataframe(scored, use_container_width=True)
                st.download_button("⬇️ Download Bulk Scoring Results CSV", scored.to_csv(index=False).encode("utf-8"), "bulk_credit_scoring_results.csv", "text/csv")
                st.download_button("⬇️ Download Bulk Scoring Results Excel", make_excel_download({"Bulk Results": scored}), "bulk_credit_scoring_results.xlsx")
        except Exception as exc:
            st.error(f"Bulk scoring failed: {exc}")
    close_card()

# -----------------------------------------------------------------------------
# PAGE: CREDIT SCORING CALCULATOR
# -----------------------------------------------------------------------------
elif section == "🧮 Credit Scoring Calculator":
    render_logo_header("Credit Scoring Calculator")
    open_card()
    st.header("🧮 Customer Credit Scoring Calculator")
    results, best_name, feature_cols = train_models_cached(df)
    best_pipeline = results[best_name]["pipeline"]
    st.markdown('<div class="gold-note">Search an existing customer ID to classify the borrower using the trained best model.</div>', unsafe_allow_html=True)
    if ID_COL in df_raw.columns:
        st.caption("Example customer IDs: " + ", ".join(df_raw[ID_COL].astype(str).head(5).tolist()))
    customer_id = st.text_input("Enter Customer ID", placeholder="Example: CUST00001")
    if st.button("🔍 Check Customer Status", use_container_width=True):
        if not customer_id.strip():
            st.warning("Please enter a Customer ID.")
        elif ID_COL not in df_raw.columns:
            st.error(f"The dataset has no `{ID_COL}` column.")
        else:
            output = predict_customer(best_pipeline, df_raw, customer_id.strip())
            if output is None:
                st.markdown("""
                <div class="info-box"><h3>💡 CREDIT STATUS: NO DATA</h3><p><b>This customer has no credit history with the bank.</b></p><hr><p><b>Risk Assessment:</b> Unknown OR New Customer</p><p><b>Recommended Action:</b></p><ul><li>Proceed with alternative credit assessment</li><li>Request business financial statements for the last 3 years</li><li>Evaluate collateral or guarantor options</li><li>Consider third-party credit bureau data</li><li>Start with a smaller loan amount for first-time borrower</li></ul></div>
                """, unsafe_allow_html=True)
            else:
                customer, pred, proba = output
                if pred == 1:
                    st.markdown(f"""
                    <div class="bad-box"><h3>🚨 CREDIT STATUS: BAD</h3><p><b>This customer is predicted as high risk / likely Bad Credit or NPL.</b></p><hr><p><b>Model Used:</b> {best_name}</p><p><b>Predicted Probability of Bad/NPL:</b> {proba:.2%}</p><p><b>Risk Assessment:</b> HIGH RISK OF DEFAULT</p><ul><li>Strongly consider loan rejection</li><li>If approved, require secured collateral, minimum 150% of loan value</li><li>Apply higher interest rate, for example +5% above base rate</li><li>Request a creditworthy guarantor with clean credit history</li><li>Recommend a shorter repayment period and close monitoring</li></ul></div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="good-box"><h3>✅ CREDIT STATUS: GOOD</h3><p><b>This customer is predicted as low risk / Good Credit.</b></p><hr><p><b>Model Used:</b> {best_name}</p><p><b>Predicted Probability of Bad/NPL:</b> {proba:.2%}</p><p><b>Risk Assessment:</b> LOW RISK OF DEFAULT</p><ul><li>Proceed with the standard loan approval process</li><li>Consider competitive interest rates</li><li>Fast-track disbursement may be considered</li><li>Offer a higher loan amount for repeat customers, subject to policy</li><li>Consider loyalty benefits or reduced processing fees</li></ul></div>
                    """, unsafe_allow_html=True)
                st.subheader("Customer Details")
                st.dataframe(customer.to_frame("Value"), use_container_width=True)
    close_card()

# -----------------------------------------------------------------------------
# PAGE: REPORTS
# -----------------------------------------------------------------------------
elif section == "📄 Loan Officer Report":
    render_logo_header("Loan Officer Report Generation")
    open_card()
    st.header("📄 Loan Officer Operational Report")
    report_tables = basic_report_tables(filtered_df, "Loan Officer Operational Report")
    for name, table in report_tables.items():
        st.subheader(name)
        st.dataframe(table, use_container_width=True)
    st.download_button("⬇️ Download Loan Officer Report Excel", make_excel_download(report_tables), "loan_officer_report.xlsx")
    close_card()

elif section == "👔 Executive Dashboard":
    render_logo_header("Executive Dashboard")
    open_card()
    st.header("👔 Executive Credit Risk Dashboard")
    total = len(filtered_df)
    bad = int((filtered_df[TARGET] == 1).sum())
    amount = filtered_df["loan_amnt"].sum() if "loan_amnt" in filtered_df.columns else 0
    avg_score = filtered_df["credit_score"].mean() if "credit_score" in filtered_df.columns else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Loans", f"{total:,}")
    c2.metric("NPL Count", f"{bad:,}")
    c3.metric("NPL Rate", f"{(bad/total*100 if total else 0):.2f}%")
    c4.metric("Total Exposure", format_money(amount))
    st.metric("Average Credit Score", f"{avg_score:.0f}" if not pd.isna(avg_score) else "-")
    tabs = st.tabs(["Portfolio", "Risk Concentration", "Executive Tables"])
    with tabs[0]:
        plot_pie(filtered_df["credit_grade"].value_counts().index, filtered_df["credit_grade"].value_counts().values, "Portfolio Good vs Bad/NPL Mix")
    with tabs[1]:
        risk_cols = [c for c in ["borrower_type", "loan_intent", "income_level", "education_or_company_size"] if c in filtered_df.columns]
        selected_risk_col = st.selectbox("Select one risk concentration view", risk_cols)
        plot_bar(group_summary(filtered_df, selected_risk_col), selected_risk_col, "npl_rate_pct", f"Executive NPL Rate by {selected_risk_col.replace('_',' ').title()}", "NPL Rate (%)")
    with tabs[2]:
        for col in ["borrower_type", "loan_intent", "income_level"]:
            if col in filtered_df.columns:
                st.subheader(col.replace("_", " ").title())
                st.dataframe(group_summary(filtered_df, col), use_container_width=True)
    close_card()

elif section == "📑 Executive Report":
    render_logo_header("Executive Report Generation")
    open_card()
    st.header("📑 Executive Report")
    tables = basic_report_tables(filtered_df, "Executive Credit Risk Report")
    model_results, best_name, _ = train_models_cached(df)
    tables["Model Summary"] = pd.DataFrame([{
        "Best Model": best_name,
        "Accuracy": round(model_results[best_name]["accuracy"], 4),
        "Precision": round(model_results[best_name]["precision"], 4),
        "Recall": round(model_results[best_name]["recall"], 4),
        "F1 Score": round(model_results[best_name]["f1"], 4),
        "AUC ROC": round(model_results[best_name]["auc"], 4),
    }])
    for name, table in tables.items():
        st.subheader(name)
        st.dataframe(table, use_container_width=True)
    st.download_button("⬇️ Download Executive Report Excel", make_excel_download(tables), "executive_credit_risk_report.xlsx")
    close_card()

# -----------------------------------------------------------------------------
# ADMIN ANALYTICS PAGES
# -----------------------------------------------------------------------------
elif section == "📋 Dataset Overview":
    render_logo_header("Dataset Overview")
    open_card()
    st.header("📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df_raw.shape[0]:,}")
    c2.metric("Original Columns", f"{df_raw.shape[1]:,}")
    c3.metric("Missing Cells", f"{int(df_raw.isna().sum().sum()):,}")
    c4.metric("Duplicate Rows", f"{int(df_raw.duplicated().sum()):,}")
    st.dataframe(df_raw.head(20), use_container_width=True)
    col_info = pd.DataFrame({"Field Name": df_raw.columns, "Description": [FIELD_DESCRIPTIONS.get(c, "Custom / derived field") for c in df_raw.columns], "Data Type": [str(t) for t in df_raw.dtypes.values], "Missing Values": df_raw.isna().sum().values, "Unique Values": df_raw.nunique(dropna=False).values})
    st.dataframe(col_info, use_container_width=True)
    close_card()

elif section == "🔍 Exploratory Data Analysis":
    render_logo_header("Exploratory Data Analysis")
    open_card()
    st.header("🔍 Exploratory Data Analysis")
    tabs = st.tabs(["Target Distribution", "Numerical", "Categorical", "Correlation"])
    with tabs[0]:
        plot_pie(df["credit_grade"].value_counts().index, df["credit_grade"].value_counts().values, "Loan Status Distribution")
    with tabs[1]:
        numeric_features = [c for c in ["income", "loan_amnt", "loan_int_rate", "loan_percent_income", "credit_history_length_years", "credit_score", "borrower_age_or_company_age"] if c in df.columns]
        selected_num = st.selectbox("Choose numerical feature", numeric_features)
        fig, ax = plt.subplots(figsize=(8.0, 3.4))
        for status, label in [(0, "Good / Performing"), (1, "Bad / NPL")]:
            ax.hist(df.loc[df[TARGET] == status, selected_num].dropna(), bins=35, alpha=0.62, label=label)
        ax.set_title(f"Distribution of {selected_num} by Loan Status", fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    with tabs[2]:
        cat_features = [c for c in ["borrower_type", "gender_or_company_entity_type", "education_or_company_size", "home_ownership_or_business_premises", "loan_intent", "previous_loan_defaults_on_file", "income_level", "loan_amount_level"] if c in df.columns]
        selected_cat = st.selectbox("Choose categorical feature", cat_features)
        data = group_summary(df, selected_cat)
        plot_bar(data, selected_cat, "npl_rate_pct", f"NPL Rate by {selected_cat.replace('_',' ').title()}", "NPL Rate (%)")
        st.dataframe(data, use_container_width=True)
    with tabs[3]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
        ax.set_title("Numerical Feature Correlation Heatmap", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    close_card()

elif section == "⚙️ Model Training & Evaluation":
    render_logo_header("Model Training and Evaluation")
    open_card()
    st.header("⚙️ Model Training & Evaluation")
    with st.spinner("Training and evaluating models..."):
        results, best_name, feature_cols = train_models_cached(df)
    res_df = pd.DataFrame([{ "Model": name, "CV F1 Mean": round(r["cv_f1_mean"], 4), "CV F1 Std": round(r["cv_f1_std"], 4), "Accuracy": round(r["accuracy"], 4), "Precision": round(r["precision"], 4), "Recall": round(r["recall"], 4), "F1 Score": round(r["f1"], 4), "AUC ROC": round(r["auc"], 4)} for name, r in results.items()]).sort_values("F1 Score", ascending=False)
    st.success(f"🏆 Best model based on F1-score: {best_name}")
    st.dataframe(res_df, use_container_width=True)
    plot_bar(res_df, "Model", "F1 Score", "Model F1 Score Comparison", "F1 Score")
    close_card()

elif section == "🎯 Best Model Results":
    render_logo_header("Best Model Results")
    open_card()
    st.header("🎯 Best Model Detailed Results")
    results, best_name, feature_cols = train_models_cached(df)
    r = results[best_name]
    best_pipeline = r["pipeline"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Model", best_name)
    c2.metric("Accuracy", f"{r['accuracy']:.3f}")
    c3.metric("Precision", f"{r['precision']:.3f}")
    c4.metric("Recall", f"{r['recall']:.3f}")
    c5.metric("AUC ROC", f"{r['auc']:.3f}")
    cm = confusion_matrix(r["y_test"], r["y_pred"])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix - {best_name}", fontweight="bold")
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["Good", "Bad/NPL"]); ax.set_yticklabels(["Good", "Bad/NPL"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=16, fontweight="bold")
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    imp = feature_importance_table(best_pipeline, feature_cols)
    if not imp.empty:
        plot_bar(imp.sort_values("importance"), "feature", "importance", "Top Model Drivers", "Importance")
    report = classification_report(r["y_test"], r["y_pred"], target_names=["Good / Performing", "Bad / NPL"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)
    close_card()

st.markdown('<p style="text-align:center; color:#FFFFFF; font-weight:700;">© 2026 Commercial Bank of Ethiopia | Role-Based AI Credit Scoring Prototype | For demonstration and research purposes</p>', unsafe_allow_html=True)

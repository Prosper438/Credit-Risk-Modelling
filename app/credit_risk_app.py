import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Amount Predictor",
    page_icon="💳",
    layout="centered"
)

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("german_credit_data.csv")

    # Drop unnamed column
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])

    # Drop missing values
    df = df.dropna(subset=['Saving accounts', 'Checking account'])

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

    # Define X and y
    y = np.log(df_encoded['Credit amount'])
    X = df_encoded.drop('Credit amount', axis=1)

    # Selected features — Risk_good excluded from user input
    significant_features = [
        'Duration',
        'Job',
        'Purpose_radio/TV',
        'Purpose_vacation/others',
        'Housing_own',
        'Risk_good',
        'Checking account_rich',
        'Sex_male',
        'Purpose_domestic appliances',
        'Checking account_moderate',
        'Saving accounts_rich'
    ]

    X_selected = X[significant_features]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test R²
    test_r2 = r2_score(y_test, model.predict(X_test))

    return model, significant_features, test_r2


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("💳 Credit Amount Predictor")
st.markdown("""
This app predicts the **credit amount** a loan applicant is likely to request
based on their financial and personal profile.

Built using **Multiple Linear Regression** on the German Credit Dataset.
""")

st.divider()

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("Loading model..."):
    try:
        model, feature_cols, test_r2 = train_model()
        st.success(f"Model loaded successfully — Test R²: {test_r2:.4f}")
    except FileNotFoundError:
        st.error("""
        ❌ **german_credit_data.csv not found.**

        Please place the CSV file in the same folder as this app and restart.
        """)
        st.stop()

st.divider()

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
st.subheader("📋 Customer Profile")
st.markdown("Fill in the applicant's details below:")

col1, col2 = st.columns(2)

with col1:
    duration = st.slider(
        "Loan Duration (months)",
        min_value=4,
        max_value=72,
        value=24,
        step=1,
        help="How many months is the loan for?"
    )

    job = st.selectbox(
        "Job Level",
        options=[0, 1, 2, 3],
        index=2,
        format_func=lambda x: {
            0: "0 — Unskilled (non-resident)",
            1: "1 — Unskilled (resident)",
            2: "2 — Skilled",
            3: "3 — Highly skilled"
        }[x],
        help="Applicant's employment skill level"
    )

    housing = st.radio(
        "Housing Status",
        options=["own", "rent", "free"],
        index=0,
        horizontal=True,
        help="Does the applicant own, rent, or live for free?"
    )

    sex = st.radio(
        "Sex",
        options=["male", "female"],
        index=0,
        horizontal=True
    )

with col2:
    checking_account = st.selectbox(
        "Checking Account Status",
        options=["little", "moderate", "rich"],
        index=1,
        help="Balance level in checking account"
    )

    saving_account = st.selectbox(
        "Saving Account Status",
        options=["little", "moderate", "quite rich", "rich"],
        index=0,
        help="Balance level in saving account"
    )

    purpose = st.selectbox(
        "Loan Purpose",
        options=[
            "car",
            "furniture/equipment",
            "radio/TV",
            "domestic appliances",
            "repairs",
            "education",
            "vacation/others",
            "business"
        ],
        index=0,
        help="What is the loan for?"
    )

st.divider()

# ─────────────────────────────────────────────
# BUILD CUSTOMER INPUT
# ─────────────────────────────────────────────
def build_customer_input(duration, job, housing, sex,
                          checking_account,
                          saving_account, purpose):
    customer = pd.DataFrame({
        'Duration':                    [duration],
        'Job':                         [job],
        'Purpose_radio/TV':            [1 if purpose == 'radio/TV' else 0],
        'Purpose_vacation/others':     [1 if purpose == 'vacation/others' else 0],
        'Housing_own':                 [1 if housing == 'own' else 0],
        'Risk_good':                   [0],  # Not inputted by user
        'Checking account_rich':       [1 if checking_account == 'rich' else 0],
        'Sex_male':                    [1 if sex == 'male' else 0],
        'Purpose_domestic appliances': [1 if purpose == 'domestic appliances' else 0],
        'Checking account_moderate':   [1 if checking_account == 'moderate' else 0],
        'Saving accounts_rich':        [1 if saving_account == 'rich' else 0],
    })

    # Reorder columns to match training order
    customer = customer[feature_cols]
    return customer


# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────
if st.button("🔍 Predict Credit Amount", use_container_width=True, type="primary"):

    # Build input
    customer = build_customer_input(
        duration, job, housing, sex,
        checking_account,
        saving_account, purpose
    )

    # Predict
    log_pred      = model.predict(customer)[0]
    original_pred = np.exp(log_pred)

    st.divider()

    # ── Result ──
    st.subheader("📊 Prediction Result")

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.metric(
            label="Predicted Credit Amount",
            value=f"DM {original_pred:,.2f}"
        )

    with col_r2:
        st.metric(
            label="Log(Credit Amount)",
            value=f"{log_pred:.4f}"
        )

    with col_r3:
        st.metric(
            label="Model Test R²",
            value=f"{test_r2:.4f}"
        )

    st.divider()

    # ── Profile Summary ──
    st.subheader("👤 Applicant Summary")

    summary_data = {
        "Feature": [
            "Loan Duration",
            "Job Level",
            "Housing",
            "Sex",
            "Checking Account",
            "Saving Account",
            "Loan Purpose"
        ],
        "Value": [
            f"{duration} months",
            f"Level {job}",
            housing.capitalize(),
            sex.capitalize(),
            checking_account.capitalize(),
            saving_account.capitalize(),
            purpose.capitalize()
        ]
    }

    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ── Interpretation ──
    st.subheader("💡 Interpretation")

    if original_pred < 2000:
        color   = "🟡"
        level   = "Low"
        advice  = "Small loan — typical for short duration or small purchases like radio/TV."
    elif original_pred < 5000:
        color   = "🟢"
        level   = "Moderate"
        advice  = "Moderate loan — consistent with medium duration and skilled applicants."
    else:
        color   = "🔴"
        level   = "High"
        advice  = "Large loan — associated with longer duration, high skill level or vacation purposes."

    st.markdown(f"""
    {color} **Predicted credit amount falls in the {level} range.**

    {advice}

    > **Note:** Predictions are in Deutsche Marks (DM) — the currency used
    > in the original German Credit Dataset.
    > Predictions are converted back from log scale using `exp()`.
    """)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown("""
    **Model:** Multiple Linear Regression

    **Dataset:** German Credit Data
    - 1,000 original records
    - 522 records after preprocessing

    **Target:** log(Credit Amount)

    **Selected Features (11):**
    - Duration
    - Job
    - Purpose (radio/TV, vacation, domestic)
    - Housing (own)
    - Checking account status
    - Sex
    - Saving account status

    **Feature Selection:**
    - f_regression (p < 0.05)
    - VIF (all < 6)

    **Performance:**
    - Train R²: 0.5059
    - Test R²:  0.5628
    - RMSE:     0.5662 (log scale)
    - No overfitting detected ✅

    ---
    > 🔮 *Risk prediction coming soon —*
    > *will be added after Logistic*
    > *Regression is implemented.*
    """)

    st.divider()
    st.markdown("""
    **Built by:** Prosper Amedu

    **GitHub:** [Prosper438](https://github.com/Prosper438)

    **Portfolio:** [prosper438.github.io](https://prosper438.github.io/prosper78.github.io/)
    """)

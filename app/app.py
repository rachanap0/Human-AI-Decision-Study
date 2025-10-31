# app/app.py
"""
Human–AI Decision Study — Guided UI with human-friendly units, descriptions, and minimal back-arrow nav
"""

import os, time, uuid, json
import pandas as pd
import streamlit as st
import joblib
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="Human–AI Decision Study", layout="centered")

# ---- minimal navigator ----
def go_to(stage: str):
    st.session_state.stage = stage
    st.rerun()

MODEL_PATH = "models/model.pkl"
LOG_PATH = "data/trials.csv"

# ---------------- Feature dictionary (name → label, unit, formatter, description) ----------------
FEATURES = {
    "MedInc": {
        "label": "Median household income",
        "unit": "$/year",
        "format": lambda v: f"${v*10_000:,.0f} / year",  # dataset is in $10,000s
        "explain": "Typical (median) yearly income for households in this neighborhood.",
    },
    "HouseAge": {
        "label": "Median house age",
        "unit": "years",
        "format": lambda v: f"{v:.0f} years",
        "explain": "Typical age of homes in the area.",
    },
    "AveRooms": {
        "label": "Average rooms per household",
        "unit": "rooms/household",
        "format": lambda v: f"{v:.2f} rooms / household",
        "explain": "Average number of rooms per household (includes bedrooms, living, etc.).",
    },
    "AveBedrms": {
        "label": "Average bedrooms per household",
        "unit": "bedrooms/household",
        "format": lambda v: f"{v:.2f} bedrooms / household",
        "explain": "Average number of bedrooms per household.",
    },
    "Population": {
        "label": "Population (district)",
        "unit": "people",
        "format": lambda v: f"{int(v):,} people",
        "explain": "Number of people living in the census district.",
    },
    "AveOccup": {
        "label": "Average household size",
        "unit": "persons/household",
        "format": lambda v: f"{v:.2f} persons / household",
        "explain": "Average number of people living in each household.",
    },
    "Latitude": {
        "label": "Latitude",
        "unit": "degrees (°)",
        "format": lambda v: f"{v:.4f}°",
        "explain": "Geographic latitude of the area.",
    },
    "Longitude": {
        "label": "Longitude",
        "unit": "degrees (°)",
        "format": lambda v: f"{v:.4f}°",
        "explain": "Geographic longitude of the area.",
    },
}
DISPLAY_COLUMNS = list(FEATURES.keys())

def dollars_from_100k(v_100k: float) -> str:
    return f"${v_100k*100_000:,.0f}"

def build_display_table(x: pd.Series) -> pd.DataFrame:
    rows = []
    for key in DISPLAY_COLUMNS:
        raw = float(x[key])
        spec = FEATURES[key]
        rows.append({
            "Field": spec["label"],
            "Value": spec["format"](raw),
            "Unit": spec["unit"],
            "What this means": spec["explain"],
        })
    return pd.DataFrame(rows)

# ---------------- Data & model loaders ----------------
@st.cache_resource
def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    df = fetch_california_housing(as_frame=True).frame.copy()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    return model, X, y

def ai_predict(model, x_series):
    return float(model.predict(pd.DataFrame([x_series]))[0])

def log_trial(row_dict):
    os.makedirs("data", exist_ok=True)
    header = not os.path.exists(LOG_PATH)
    pd.DataFrame([row_dict]).to_csv(LOG_PATH, mode="a", header=header, index=False)

# ---------------- App state ----------------
try:
    model, X, y = load_model_and_data()
except Exception as e:
    st.error(f"Could not load model or data: {e}")
    st.stop()

if "stage" not in st.session_state:
    st.session_state.stage = "intro"  # intro → round1 → round2 → done
if "uid" not in st.session_state:
    st.session_state.uid = str(uuid.uuid4())[:8]
if "tasks" not in st.session_state:
    r1 = X.sample(1, random_state=uuid.uuid4().int % (2**32 - 1)).iloc[0]
    r2 = X.sample(1, random_state=uuid.uuid4().int % (2**32 - 1)).iloc[0]
    st.session_state.tasks = {"1": {"x": r1, "y": float(y.loc[r1.name])},
                              "2": {"x": r2, "y": float(y.loc[r2.name])}}

uid = st.session_state.uid

# ---------------- simple progress indicator ----------------
stage2idx = {"intro": 0, "round1": 1, "round2": 2, "done": 3}
st.progress(stage2idx.get(st.session_state.stage, 0) / 3.0)

# ---------------- Screens ----------------
if st.session_state.stage == "intro":
    st.title("🏠 Human–AI Decision Study")
    st.markdown(
        """
**What you’re doing:** Estimate the **median home value** for a California neighborhood.

**How it works**
- **Round 1:** You estimate on your own (no AI).
- **Round 2:** You’ll see an **AI suggestion** and can use/ignore it.

**What the numbers mean**
- We show neighborhood stats like income, house age, rooms, population, and location — all in **plain units**.
        """
    )
    if st.button("🚀 Start Experiment"):
        go_to("round1")

elif st.session_state.stage == "round1":
    # minimalist back arrow (to Intro)
    col_back, col_title = st.columns([1, 20])
    with col_back:
        if st.button("←", help="Back", use_container_width=True, key="back_intro_arrow"):
            st.session_state.pop("t0_r1", None)  # optional: reset timer
            go_to("intro")
    with col_title:
        st.header("Round 1 — Predict Without AI")
        st.write("Look at this neighborhood’s information and estimate the **median home value**. Values are based on real California census data.")

    task = st.session_state.tasks["1"]
    x, y_true_100k = task["x"], task["y"]

    st.markdown("### Neighborhood Information")
    # Use static table so there are no scrollbars or extra blank rows
    st.table(build_display_table(x).set_index("Field"))

    with st.expander("What each field means"):
        for key in DISPLAY_COLUMNS:
            spec = FEATURES[key]
            st.markdown(f"- **{spec['label']}** ({spec['unit']}): {spec['explain']}")

    pred_dollars = st.number_input("Your estimate (in dollars)", min_value=0.0, step=1_000.0, format="%.0f")
    conf = st.slider("How confident are you?", 0, 100, 60)

    if "t0_r1" not in st.session_state:
        st.session_state.t0_r1 = time.time()

    _, submit_col = st.columns([5, 1])
    with submit_col:
        if st.button("Submit", use_container_width=True):
            dur = round(time.time() - st.session_state.t0_r1, 3)
            human_100k = (pred_dollars or 0.0) / 100_000.0
            abs_err = abs(human_100k - y_true_100k)

            table_logged = build_display_table(x)  # rebuild to log exact view
            log_trial({
                "uid": uid, "round_idx": 1, "condition": "no_ai", "ts": int(time.time()),
                "duration_sec": dur, "feat_json_displayed": json.dumps(table_logged.to_dict(orient="records")),
                "human_pred_dollars": float(pred_dollars or 0.0), "human_pred_100k": float(human_100k),
                "ai_pred_dollars": None, "ai_pred_100k": None,
                "y_true_dollars": float(y_true_100k*100_000), "y_true_100k": float(y_true_100k),
                "confidence_0_100": int(conf), "abs_error_100k": float(abs_err)
            })
            go_to("round2")

elif st.session_state.stage == "round2":
    # minimalist back arrow (to Round 1)
    col_back, col_title = st.columns([1, 20])
    with col_back:
        if st.button("←", help="Back", use_container_width=True, key="back_r1_arrow"):
            st.session_state.pop("t0_r2", None)  # optional: reset timer
            go_to("round1")
    with col_title:
        st.header("Round 2 — Predict With AI Assistance")
        st.write("You’ll now see a similar neighborhood **plus** an AI model’s suggested estimate. Use it, ignore it, or adjust your own number.")

    task = st.session_state.tasks["2"]
    x, y_true_100k = task["x"], task["y"]
    ai_pred_100k = ai_predict(model, x)

    st.markdown("### Neighborhood Information")
    st.table(build_display_table(x).set_index("Field"))  # static table → no blank rows
    st.info(f"🤖 AI suggests: **{dollars_from_100k(ai_pred_100k)}**")

    with st.expander("What each field means"):
        for key in DISPLAY_COLUMNS:
            spec = FEATURES[key]
            st.markdown(f"- **{spec['label']}** ({spec['unit']}): {spec['explain']}")

    pred_dollars = st.number_input("Your final estimate (in dollars)", min_value=0.0, step=1_000.0, format="%.0f")
    conf = st.slider("How confident are you?", 0, 100, 60)

    if "t0_r2" not in st.session_state:
        st.session_state.t0_r2 = time.time()

    _, submit_col = st.columns([5, 1])
    with submit_col:
        if st.button("Submit", use_container_width=True):
            dur = round(time.time() - st.session_state.t0_r2, 3)
            human_100k = (pred_dollars or 0.0) / 100_000.0
            abs_err = abs(human_100k - y_true_100k)

            table_logged = build_display_table(x)
            log_trial({
                "uid": uid, "round_idx": 2, "condition": "with_ai", "ts": int(time.time()),
                "duration_sec": dur, "feat_json_displayed": json.dumps(table_logged.to_dict(orient="records")),
                "human_pred_dollars": float(pred_dollars or 0.0), "human_pred_100k": float(human_100k),
                "ai_pred_dollars": float(ai_pred_100k*100_000), "ai_pred_100k": float(ai_pred_100k),
                "y_true_dollars": float(y_true_100k*100_000), "y_true_100k": float(y_true_100k),
                "confidence_0_100": int(conf), "abs_error_100k": float(abs_err)
            })
            go_to("done")

else:
    st.balloons()
    st.success("🎉 Thank you! You’ve completed both rounds.")
    st.caption("You can close this tab or click below to run again.")
    if st.button("Run again"):
        for k in ["stage", "tasks", "t0_r1", "t0_r2"]:
            st.session_state.pop(k, None)
        go_to("intro")

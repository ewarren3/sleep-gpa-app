import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Sleep & GPA what-if", page_icon="🌙", layout="wide")

# =====================================================================
# Load the trained model + the metadata exported from Colab
# =====================================================================
@st.cache_resource
def load_artifacts():
    booster = xgb.Booster()
    booster.load_model("model.json")
    with open("app_meta.json") as f:
        meta = json.load(f)
    explainer = shap.TreeExplainer(booster)
    return booster, meta, explainer


booster, meta, explainer = load_artifacts()

FEATURES = meta["feature_names"]
MEDIANS = meta["medians"]
BRACKETS = meta["bracket_order"]
SEMESTERS = meta["semester_order"]
GENDER_MAP = meta["gender_map"]
UNIVERSITIES = meta["university_categories"]
COHORTS = meta["cohort_categories"]
UW_MEAN = meta["uw_term_units"]["mean"]
UW_SD = meta["uw_term_units"]["sd"]
TEST_RMSE = meta.get("test_rmse")

# Features hidden from the charts (their effect is folded into the baseline)
HIDE = [c for c in ["first_generation", "underrepresented"] if c in FEATURES]

# The dataset's sleep_midpoint_minutes runs 60 higher than the clock-time midpoint
MIDPOINT_OFFSET = 60

PRETTY = {
    "prior_gpa": "Prior-term GPA",
    "avg_sleep_hours": "Avg nightly sleep (hours)",
    "avg_sleep_minutes": "Avg nightly sleep (minutes)",
    "daytime_sleep_minutes": "Daytime sleep (minutes)",
    "sleep_midpoint_minutes": "Sleep midpoint (minutes)",
    "bedtime_variability": "Bedtime variability",
    "nights_tracked_fraction": "Fraction of nights tracked",
    "under_6h_sleep": "Sleeps under 6 hrs (1 = yes)",
    "sleep_bracket": "Sleep duration bracket",
    "term_units": "Term units",
    "term_load_z": "Course load (z-score)",
    "university": "University",
    "semester": "Semester",
    "cohort_code": "Cohort",
    "gender": "Gender",
    "study": "Study",
}
# Integer-coded categoricals: the code means nothing to a reader, so show only the name
NO_VALUE = {"university", "semester", "cohort_code", "gender", "sleep_bracket", "study"}


# =====================================================================
# Turning friendly inputs into the model's features
# =====================================================================
def bracket_code(h):
    if h < 5.5:
        label = "<5.5h"
    elif h < 6:
        label = "5.5-6h"
    elif h < 6.5:
        label = "6-6.5h"
    elif h < 7:
        label = "6.5-7h"
    elif h < 7.5:
        label = "7-7.5h"
    else:
        label = "7.5h+"
    return BRACKETS[label]


def hour_label(h):
    """Hours after midnight (negative = evening) -> '1:30 AM' style label."""
    t = h % 24
    hh = int(t)
    mm = int(round((t - hh) * 60))
    suffix = "AM" if hh < 12 else "PM"
    return f"{(hh % 12) or 12}:{mm:02d} {suffix}"


BED_OPTIONS = {hour_label(h): float(h) for h in np.arange(-3, 5.01, 0.5)}


def build_row(inp):
    # Start from the typical (median) student, then overwrite what the user told us
    row = {f: (np.nan if MEDIANS.get(f) is None else MEDIANS[f]) for f in FEATURES}

    hrs = inp["sleep_hours"]
    midpoint_clock_h = inp["bedtime"] + hrs / 2  # hours after midnight
    credits = inp["credits"]

    derived = {
        "avg_sleep_hours": hrs,
        "avg_sleep_minutes": hrs * 60,
        "under_6h_sleep": int(hrs < 6),
        "sleep_bracket": bracket_code(hrs),
        "sleep_midpoint_minutes": midpoint_clock_h * 60 + MIDPOINT_OFFSET,
        "bedtime_variability": inp["window"] ** 2 / 12,  # variance if bedtime is spread evenly over the window
        "prior_gpa": inp["prior_gpa"],
        "daytime_sleep_minutes": inp["daytime"],
        "nights_tracked_fraction": inp["tracked"],
        "term_units": credits,
        "term_load_z": (credits - UW_MEAN) / UW_SD,
    }

    if inp["university"] in UNIVERSITIES:
        derived["university"] = UNIVERSITIES.index(inp["university"])
    elif inp["university"] == "Other / not listed":
        derived["university"] = -1
    if inp["cohort"] in COHORTS:
        derived["cohort_code"] = COHORTS.index(inp["cohort"])
    elif inp["cohort"] == "Other / not listed":
        derived["cohort_code"] = -1
    if inp["semester"] in SEMESTERS:
        derived["semester"] = SEMESTERS[inp["semester"]]
    derived["gender"] = GENDER_MAP.get(inp["gender"], np.nan)
    derived["first_generation"] = {"Yes": 1, "No": 0}.get(inp["first_gen"], np.nan)
    derived["underrepresented"] = {"Yes": 1, "No": 0}.get(inp["underrep"], np.nan)

    for k, v in derived.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row])[FEATURES].astype(float)


def predict(inp):
    return float(booster.predict(xgb.DMatrix(build_row(inp)))[0])


# =====================================================================
# Sidebar inputs
# =====================================================================
st.sidebar.header("Your inputs")

sleep_hours = st.sidebar.slider("Average nightly sleep (hours)", 4.0, 10.0, 7.0, 0.25)
bedtime_label = st.sidebar.select_slider(
    "Typical bedtime", options=list(BED_OPTIONS), value="1:00 AM"
)
window = st.sidebar.slider(
    "Bedtime regularity: your bedtime usually falls within a window of about (hours)",
    0.0, 4.0, 1.0, 0.25,
    help="0 = same time every night. 2 = bedtime wanders across about a 2-hour window.",
)
prior_gpa = st.sidebar.slider("Prior-term GPA", 1.5, 4.0, 3.5, 0.05)
credits = st.sidebar.number_input("Credits this term", min_value=6, max_value=24, value=16, step=1)

with st.sidebar.expander("Advanced options"):
    daytime = st.slider(
        "Daytime sleep (minutes per day)", 0, 120, int(MEDIANS.get("daytime_sleep_minutes") or 35)
    )
    tracked = st.slider(
        "Fraction of nights tracked", 0.2, 1.0, float(round(MEDIANS.get("nights_tracked_fraction") or 0.93, 2)), 0.01
    )
    university = st.selectbox("University", ["Not specified"] + UNIVERSITIES + ["Other / not listed"])
    cohort = st.selectbox("Cohort", ["Not specified"] + COHORTS + ["Other / not listed"])
    semester = st.selectbox("Semester", ["Not specified"] + list(SEMESTERS))
    gender = st.selectbox("Gender", ["Prefer not to say"] + list(GENDER_MAP))
    first_gen = st.selectbox("First-generation student", ["Prefer not to say", "No", "Yes"])
    underrep = st.selectbox("Underrepresented group", ["Prefer not to say", "No", "Yes"])

dark = st.sidebar.checkbox("Dark charts", value=False)

inp = {
    "sleep_hours": float(sleep_hours),
    "bedtime": BED_OPTIONS[bedtime_label],
    "window": float(window),
    "prior_gpa": float(prior_gpa),
    "credits": float(credits),
    "daytime": float(daytime),
    "tracked": float(tracked),
    "university": university,
    "cohort": cohort,
    "semester": semester,
    "gender": gender,
    "first_gen": first_gen,
    "underrep": underrep,
}


# =====================================================================
# Charts
# =====================================================================
def theme(dark_mode):
    if dark_mode:
        return dict(bg="#0E1117", fg="#F2F2F2", muted="#9A9A9A", up="#4CD18A", down="#FF6B6B", grid="#FFFFFF")
    return dict(bg="#FFFFFF", fg="#1A1A1A", muted="#666666", up="#2E8B57", down="#C0392B", grid="#000000")


def waterfall_fig(sv, dark_mode=False, top_n=10):
    t = theme(dark_mode)
    vals_all = sv.values[0]
    data_all = sv.data[0]
    base = float(np.ravel(sv.base_values)[0])

    hide_idx = [FEATURES.index(c) for c in HIDE]
    keep_idx = [i for i in range(len(FEATURES)) if i not in hide_idx]
    base_adj = base + float(vals_all[hide_idx].sum()) if hide_idx else base

    keep_names = [FEATURES[i] for i in keep_idx]
    vals = vals_all[keep_idx]
    data = data_all[keep_idx]
    pred = base_adj + vals.sum()

    order = np.argsort(-np.abs(vals))
    top, rest = order[:top_n], order[top_n:]

    items = []
    for i in top:
        name = PRETTY.get(keep_names[i], keep_names[i].replace("_", " ").title())
        if keep_names[i] in NO_VALUE or pd.isna(data[i]):
            label = name
        else:
            v = float(data[i])
            label = f"{name} = {v:.0f}" if v.is_integer() else f"{name} = {v:.2f}"
        items.append((label, float(vals[i])))
    if len(rest) > 0:
        items.append((f"{len(rest)} other features", float(vals[rest].sum())))

    items = sorted(items, key=lambda x: abs(x[1]))
    items = sorted(items, key=lambda x: not x[0].endswith("other features"))

    fig, ax = plt.subplots(figsize=(9, 5.8), facecolor=t["bg"])
    ax.set_facecolor(t["bg"])

    cum = base_adj
    for y, (label, v) in enumerate(items):
        ax.barh(y, v, left=cum, color=t["up"] if v >= 0 else t["down"], height=0.6, edgecolor="none")
        ax.text(
            cum + v + (0.004 if v >= 0 else -0.004), y, f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9, color=t["fg"],
        )
        cum += v

    ax.axvline(base_adj, color=t["muted"], linestyle="--", linewidth=1)
    ax.axvline(pred, color=t["fg"], linestyle="-", linewidth=1.2)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([lbl for lbl, _ in items], fontsize=10, color=t["fg"])

    deltas = [v for _, v in items]
    lo = min(base_adj, pred, base_adj + min(0, min(deltas))) - 0.15
    hi = max(base_adj, pred) + max(0, max(deltas)) + 0.15
    ax.set_xlim(lo, hi)

    ax.set_xlabel("Predicted term GPA", fontsize=11, color=t["fg"])
    ax.set_title(f"Predicted term GPA: {pred:.2f}", fontsize=16, fontweight="bold", color=t["fg"], pad=26)
    fig.text(
        0.5, 0.90,
        f"Average prediction: {base_adj:.2f}   |   green raises predicted GPA, red lowers it (GPA points)",
        ha="center", fontsize=9, color=t["muted"],
    )
    ax.text(base_adj, len(items) - 0.35, "Average", ha="center", va="bottom", fontsize=8, color=t["muted"])
    ax.text(pred, len(items) - 0.35, "You", ha="center", va="bottom", fontsize=8, color=t["fg"])

    ax.tick_params(colors=t["fg"])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(t["muted"])
    ax.grid(axis="x", color=t["grid"], alpha=0.12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


SWEEPS = {
    "Average nightly sleep (hours)": ("sleep_hours", np.arange(4.0, 10.01, 0.25)),
    "Bedtime regularity window (hours)": ("window", np.arange(0.0, 4.01, 0.25)),
    "Credits this term": ("credits", np.arange(6.0, 24.1, 1.0)),
    "Prior-term GPA": ("prior_gpa", np.arange(1.5, 4.001, 0.1)),
}


def sweep_fig(inp, key, grid, dark_mode=False):
    t = theme(dark_mode)
    preds = []
    for g in grid:
        trial = dict(inp)
        trial[key] = float(g)
        preds.append(predict(trial))
    now = predict(inp)

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=t["bg"])
    ax.set_facecolor(t["bg"])
    ax.plot(grid, preds, color=t["up"], linewidth=2.2)
    ax.scatter([inp[key]], [now], color=t["fg"], zorder=5, s=55)
    ax.axhline(now, color=t["muted"], linestyle="--", linewidth=1)
    ax.set_ylabel("Predicted term GPA", color=t["fg"])
    ax.tick_params(colors=t["fg"])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(t["muted"])
    ax.grid(color=t["grid"], alpha=0.1)
    fig.tight_layout()
    return fig, preds


# =====================================================================
# Main page
# =====================================================================
st.title("🌙 Sleep & term GPA: what-if explorer")
st.caption(
    "Change the inputs on the left and see how the model's predicted term GPA moves. "
    "This is a prediction from data on first-year students, not a guarantee or a causal claim."
)

df_row = build_row(inp)
sv = explainer(df_row)
pred = float(booster.predict(xgb.DMatrix(df_row))[0])
base_adj = float(np.ravel(sv.base_values)[0]) + (
    float(sv.values[0][[FEATURES.index(c) for c in HIDE]].sum()) if HIDE else 0.0
)

c1, c2, c3 = st.columns(3)
c1.metric("Predicted term GPA", f"{min(max(pred, 0.0), 4.0):.2f}", delta=f"{pred - base_adj:+.2f} vs. average student")
c2.metric("Average prediction", f"{base_adj:.2f}")
if TEST_RMSE:
    c3.metric("Typical error", f"± {TEST_RMSE:.2f} GPA pts", help="Test-set RMSE of the model")

tab1, tab2, tab3 = st.tabs(["Why this prediction", "What if I change something?", "About this model"])

with tab1:
    st.pyplot(waterfall_fig(sv, dark_mode=dark))
    st.caption("Each bar shows how much a feature pushed your predicted GPA above (green) or below (red) the average.")

with tab2:
    choice = st.selectbox("Vary:", list(SWEEPS))
    key, grid = SWEEPS[choice]
    fig, preds = sweep_fig(inp, key, grid, dark_mode=dark)
    st.pyplot(fig)
    st.caption(
        f"Holding everything else fixed, predicted GPA ranges from {min(preds):.2f} to {max(preds):.2f} "
        f"across this range. The white dot is your current input."
    )

with tab3:
    st.markdown(
        """
**Data:** 634 first-year students at three universities, with Fitbit-measured sleep for about a month early in
the spring term (2016 to 2019) and registrar GPAs.

**Model:** gradient-boosted trees (XGBoost) predicting end-of-term GPA.

**Important limits**
- The data are observational. The model finds associations between sleep and grades; it cannot show that changing
  your sleep would change your GPA.
- Prior-term GPA is by far the strongest predictor, and the model is only moderately accurate overall.
- Two inputs are converted approximately: *bedtime regularity* (the dataset's exact formula is not documented) and
  *credits* (compared against the University of Washington samples' credit scale).
- Sliders far outside the range of the training data (for example, under 5 hours of sleep) are extrapolations.
"""
    )

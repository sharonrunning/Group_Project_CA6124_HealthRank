"""
app.py  (Member B) - HealthRank interactive demo

Run from the project root:
    streamlit run app.py

Two pages in a side navigation:
  - Meals for you   : end-user view (meal cards, details dialog), model weight fixed at the trained λ
  - Model insights  : score breakdown, held-out test matches and history comparison for the teaching team
"""
import html
import os

import numpy as np
import pandas as pd
import streamlit as st

from src.integration import Recommender

st.set_page_config(page_title="HealthRank", page_icon="🥗", layout="wide")

# palette: brand teal on light surfaces (text colours checked for >= 4.5:1 contrast)
NAVY, MUTED, TEAL, TEAL_DARK, APRICOT = "#173F46", "#5F6E75", "#258D87", "#1F7A75", "#E8A477"
GREEN, GREEN_SOFT, PREF = TEAL_DARK, "#D4EAE5", "#8EA3AE"   # used by the Model insights page
CAT_ICON = {"appetizer": "🥗", "main_dish": "🍲", "dessert": "🍰"}
CAT_LABEL = {"appetizer": "Appetizer", "main_dish": "Main dish", "dessert": "Dessert"}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    html, body, .stMarkdown, button, input, label {{font-family: 'DM Sans', sans-serif;}}
    .block-container {{padding-top: 2.6rem; max-width: 1180px;}}
    /* sidebar */
    [data-testid="stSidebar"] {{background: #EAF2F3; border-right: 1px solid #DCE7E9;}}
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] * {{color: {NAVY} !important;}}
    [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{border-radius: 10px; padding: 8px 12px;}}
    div[class*="st-key-nav_on"] a {{background: #D4EAE5 !important; box-shadow: inset 3px 0 0 {TEAL};}}
    div[class*="st-key-nav_on"] a * {{font-weight: 700 !important;}}
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{background: #FFFFFF !important;}}
    .brand {{display:flex; gap:12px; align-items:center; margin: 4px 0 20px 0;}}
    .brand-mark {{width:40px; height:40px; border-radius:12px; background:{TEAL}; position:relative; flex:0 0 40px;}}
    .brand-mark::after {{content:""; position:absolute; right:7px; top:7px; width:12px; height:12px;
                         border-radius:50%; background:{APRICOT};}}
    .brand-name {{font-size:1.35rem; font-weight:700; color:{NAVY}; line-height:1;}}
    .brand-tag {{font-size:0.85rem; color:{MUTED}; margin-top:4px;}}
    .side-note {{font-size:0.875rem; color:{MUTED}; line-height:1.5; margin-top:14px;}}
    /* page */
    .page-title {{font-size:2.1rem; font-weight:700; color:{NAVY}; line-height:1.15;}}
    .page-sub {{color:{MUTED}; font-size:0.95rem; margin:8px 0 6px 0; max-width: 760px;}}
    .chips {{display:flex; gap:8px; flex-wrap:wrap; margin: 10px 0 18px 0;}}
    .chip {{background:#EEF1F0; color:{NAVY}; border-radius:999px; padding:4px 12px; font-size:0.875rem;}}
    /* meal cards: fixed text blocks so bars, reasons and buttons line up across a row */
    div[class*="st-key-card_"] {{background:#FFFFFF; border:1px solid #E3E8E8 !important; border-radius:14px;
                                box-shadow: 0 1px 2px rgba(23,63,70,0.05);}}
    .c-top {{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;}}
    .c-num {{font-size:0.82rem; font-weight:700; color:{TEAL_DARK}; letter-spacing:.08em;}}
    .c-score {{font-size:0.875rem; color:{MUTED};}}
    .c-score b {{color:{NAVY};}}
    .c-title {{font-weight:700; font-size:1.15rem; color:{NAVY}; line-height:1.3; height:2.6em; overflow:hidden;
               display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;}}
    .c-side {{font-size:0.875rem; color:{MUTED}; line-height:1.5; white-space:nowrap; overflow:hidden;
              text-overflow:ellipsis; margin-top:2px;}}
    .c-side b {{color:{NAVY}; font-weight:500;}}
    .c-track {{height:6px; border-radius:3px; background:#E6ECEC; overflow:hidden; margin:14px 0 4px 0;}}
    .c-fill {{height:100%; background:{TEAL};}}
    .c-why {{font-size:0.9rem; color:{NAVY}; margin:10px 0 0 0; height:2.9em; line-height:1.45; overflow:hidden;
             display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;}}
    div[class*="st-key-view_"] button p {{color:{TEAL_DARK} !important; font-weight:700;}}
    /* details dialog */
    .d-row {{display:flex; justify-content:space-between; gap:12px; padding:12px 0; border-bottom:1px solid #E6ECEC;}}
    .d-cat {{font-size:0.75rem; letter-spacing:.08em; text-transform:uppercase; color:{MUTED};}}
    .d-name {{font-weight:700; color:{NAVY};}}
    .d-meta {{font-size:0.875rem; color:{MUTED}; text-align:right; white-space:nowrap;}}
    .eyebrow {{font-size:0.78rem; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; margin:22px 0 8px 0;}}
    /* Model insights page */
      .hr-sub {{color: #6b726d; margin-top: -0.6rem; margin-bottom: 1.2rem;}}
      .hr-badge {{display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.85rem;
                 font-weight: 600; margin-left: 8px;}}
      .hr-dish {{margin: 3px 0; font-size: 0.97rem;}}
      .hr-dish span {{color: #6b726d; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em;
                     display: inline-block; width: 110px;}}
      .hr-legend {{font-size: 0.8rem; color: #6b726d;}}
      .hr-reason {{font-size: 0.88rem; color: #3a413c; margin: 3px 0;}}
      .hr-rank {{font-size: 1.6rem; font-weight: 700; color: #1F7A75; line-height: 1.1;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource  # model + data are loaded once per server, not on every interaction
def get_recommender():
    return Recommender("MealRec+H")


try:
    rec = get_recommender()
except Exception as e:  # show the real error instead of silently falling back to a placeholder
    st.error(f"Could not load the recommender model: **{type(e).__name__}: {e}**")
    st.exception(e)
    st.info("Check that `models/checkpoints/health_aware_bpr.pt` exists and that all packages in "
            "`requirements.txt` (including torch) are installed.")
    st.stop()
data = rec.data


@st.cache_data
def course_health():
    return pd.read_csv(os.path.join("data", "processed", "MealRec+H", "course_health.csv")).set_index("course_id")


def health_badge(h):
    """Badge whose strength follows the health score (higher = healthier)."""
    if h >= 0.5:
        bg, fg, word = GREEN, "#ffffff", "higher"
    elif h >= 0.35:
        bg, fg, word = GREEN_SOFT, "#155f55", "moderate"
    else:
        bg, fg, word = "#eceee9", "#4a514c", "lower"
    return f'<span class="hr-badge" style="background:{bg};color:{fg}">Health {h:.2f} · {word}</span>'


# ---------------- shared state ----------------
test_users = sorted(data.test_items)
only_test = st.session_state.get("only_test", True)
pool = test_users if only_test else list(range(data.n_users))
if st.session_state.get("user_id") not in pool:
    st.session_state["user_id"] = pool[0]
st.session_state.setdefault("k", 9)


def get_recs():
    return rec.recommend_with_details(int(st.session_state["user_id"]), int(st.session_state["k"]))


def by_cat(r):
    return {c["category"]: c for c in r["courses"]}


# one plain sentence per card; the most specific reason first
REASON_PRIORITY = ("Healthier", "Includes", "Similar", "Popular", "Based")


def main_reason(reasons):
    for key in REASON_PRIORITY:
        for x in reasons:
            if x.startswith(key):
                return x
    return reasons[0] if reasons else ""


# ---------------- Meals for you ----------------
@st.dialog("Meal details", width="large")
def meal_dialog(r):
    cats = by_cat(r)
    ch = course_health()
    st.markdown(f'<div class="page-title" style="font-size:1.6rem">{html.escape(cats["main_dish"]["name"])}</div>'
                f'<div class="page-sub">Meal {r["meal_id"]} · appetizer + main dish + dessert · '
                f'nutrition score <b>{r["health_score"]:.2f}</b> / 1</div>', unsafe_allow_html=True)
    rows_html = ""
    for cat in ("appetizer", "main_dish", "dessert"):
        if cat not in cats:
            continue
        c = cats[cat]
        rating = f'★ {c["rating"]:.1f} · {c["reviews"]:,} reviews' if c.get("rating") else ""
        hl = ""
        if c["course_id"] in ch.index:
            hrow = ch.loc[c["course_id"]]
            hl = f'<br>FSA {hrow.fsa:.0f} (lower = healthier) · WHO {hrow.who:.0f} (higher = healthier)'
        rows_html += (f'<div class="d-row"><div><div class="d-cat">{CAT_LABEL[cat]}</div>'
                      f'<div class="d-name">{html.escape(c["name"])}</div></div>'
                      f'<div class="d-meta">{rating}{hl}</div></div>')
    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Why this meal</div>' +
                "".join(f"<div style='margin:4px 0;color:{NAVY}'>• {html.escape(x)}</div>" for x in r["user_reasons"]),
                unsafe_allow_html=True)


def meal_card(r):
    cats = by_cat(r)
    h = r["health_score"]
    sides = "".join(
        f'<div class="c-side" title="{html.escape(cats[cat]["name"])}">{CAT_LABEL[cat]} · '
        f'<b>{html.escape(cats[cat]["name"])}</b></div>'
        for cat in ("appetizer", "dessert") if cat in cats
    )
    main = cats.get("main_dish")
    with st.container(border=True, key=f"card_{r['meal_id']}"):
        st.markdown(
            f'<div class="c-top"><span class="c-num">MEAL {r["rank"]:02d}</span>'
            f'<span class="c-score">Nutrition <b>{h:.2f}</b> / 1</span></div>'
            f'<div class="c-title">{html.escape(main["name"]) if main else "Meal " + str(r["meal_id"])}</div>'
            f'{sides}'
            f'<div class="c-track"><div class="c-fill" style="width:{100 * h:.0f}%"></div></div>'
            f'<div class="c-why">{html.escape(main_reason(r["user_reasons"]))}</div>',
            unsafe_allow_html=True,
        )
        with st.container(key=f"view_{r['meal_id']}"):
            if st.button("View meal  →", key=f"btn_{r['meal_id']}", type="tertiary"):
                meal_dialog(r)


def page_meals():
    profile, rows = get_recs()
    avg = float(np.mean([r["health_score"] for r in rows])) if rows else float("nan")
    st.markdown('<div class="page-title">Meals picked for you</div>'
                '<div class="page-sub">Complete meals — appetizer, main dish and dessert — recommended based on this '
                'profile’s interaction history, with a nudge towards healthier options.</div>'
                f'<div class="chips"><span class="chip">{len(rows)} meals</span>'
                f'<span class="chip">Average nutrition score {avg:.2f} / 1</span></div>', unsafe_allow_html=True)
    with st.expander("About the nutrition score"):
        st.markdown(
            "The **nutrition score** (0 to 1, higher = healthier) is defined by our project. It combines the two "
            "health scores provided by the MealRec+ dataset: **FSA** (UK Food Standards Agency traffic-light "
            "criteria, lower = healthier) and **WHO** (World Health Organization guidelines met, higher = healthier). "
            "Both are rescaled to 0–1 and averaged, with FSA flipped so that higher always means healthier. "
            "It describes the recipes' nutrition only and is not personal dietary advice."
        )
    # one st.columns row per three cards, so cards keep rank order when columns stack on a phone
    for i in range(0, len(rows), 3):
        cols = st.columns(3, gap="medium")
        for col, r in zip(cols, rows[i:i + 3]):
            with col:
                meal_card(r)


# ---------------- Model insights ----------------
def page_insights():
    profile, rows = get_recs()
    k = int(st.session_state["k"])
    st.markdown('<div class="page-title">Model insights</div>'
                '<div class="page-sub">How each recommendation is scored, checked against the user’s held-out test '
                'interactions. Intended for the teaching team.</div>', unsafe_allow_html=True)
    with st.expander("Technical settings"):
        st.checkbox("Only users with test data", key="only_test", value=True,
                    help="These users have held-out interactions, so recommendations can be checked "
                         "against data that was not used for training.")
        if rec.is_placeholder:
            st.warning(f"Model: **{rec.model_name}** - no model file from Member A found.")
        else:
            st.markdown(f"Model: **{rec.model_name}**")
            if rec.has_score_breakdown:
                st.caption("Score = personalised preference (BPR-MF) + λ × health score, where the model rescales "
                           "health_score to [0, 1] internally before multiplying by λ.")
    rec_health = float(np.mean([r["health_score"] for r in rows])) if rows else float("nan")
    hits = sum(r["in_test_set"] for r in rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meals in history", profile["n_train_meals"], border=True, help="Training interactions of this user")
    c2.metric("Avg health · history", f"{profile['mean_health_score']:.2f}", border=True)
    c3.metric(f"Avg health · top-{k}", f"{rec_health:.2f}", border=True,
              delta=f"{rec_health - profile['mean_health_score']:+.2f} vs history" if rows else None)
    c4.metric("Matches with test set", f"{hits} / {len(profile['test_meals'])}", border=True,
              help="Recommended meals that are among the user's held-out test interactions")

    # ---------------- recommendation cards ----------------
    st.subheader("Recommended meals")

    for r in rows:
        with st.container(border=True):
            left, mid, right = st.columns([0.6, 4.4, 4.0])
            with left:
                st.markdown(f'<div class="hr-rank">#{r["rank"]}</div>', unsafe_allow_html=True)
                if r["in_test_set"]:
                    st.markdown("✅", help="This meal is one of the user's held-out test interactions")
            with mid:
                title = f'<b>Meal {r["meal_id"]}</b>{health_badge(r["health_score"])}'
                dishes = "".join(
                    f'<div class="hr-dish"><span>{CAT_ICON.get(c["category"], "")} '
                    f'{CAT_LABEL.get(c["category"], c["category"])}</span>{html.escape(c["name"])}</div>'
                    for c in r["courses"]
                )
                st.markdown(title + dishes, unsafe_allow_html=True)
                st.markdown(f'<div class="hr-legend">FSA {r["fsa"]:.2f} (lower = healthier) · '
                            f'WHO {r["who"]:.2f} (higher = healthier)</div>', unsafe_allow_html=True)
            with right:
                if r["preference"] is not None:
                    st.markdown(
                        f'<div class="hr-legend">Model score</div>'
                        f'<div style="font-size:0.95rem;margin:2px 0 6px 0"><b>{r["score"]:.3f}</b> = '
                        f'preference {r["preference"]:.3f} + health bonus {r["health_bonus"]:.3f}</div>',
                        unsafe_allow_html=True,
                    )
                    reasons = r["reasons"][1:]  # first reason repeats the score split shown above
                else:
                    st.markdown(f'<div class="hr-legend">Model score <b>{r["score"]:.3f}</b></div>',
                                unsafe_allow_html=True)
                    reasons = r["reasons"]
                st.markdown("".join(f'<div class="hr-reason">• {html.escape(x)}</div>' for x in reasons),
                            unsafe_allow_html=True)

    # ---------------- health comparison ----------------
    st.subheader("Healthiness: history vs recommendations")
    hist_h = data.health[profile["history"]] if profile["history"] else np.array([])
    chart = pd.DataFrame({
        "History (train)": pd.Series(hist_h),
        f"Top-{k} recommended": pd.Series([r["health_score"] for r in rows]),
    })
    bins = np.linspace(0, 1, 11)
    counts = pd.DataFrame({
        col: pd.cut(chart[col].dropna(), bins, include_lowest=True).value_counts(normalize=True).sort_index()
        for col in chart.columns
    })
    counts.index = [f"{b.left:.1f}-{b.right:.1f}" for b in counts.index]
    st.bar_chart(counts, color=[PREF, GREEN], stack=False, x_label="Health score", y_label="Share of meals")

    # ---------------- details ----------------
    with st.expander("Full recommendation table"):
        table = pd.DataFrame([{
            "Rank": r["rank"],
            "Meal": r["meal_id"],
            "Appetizer": next((c["name"] for c in r["courses"] if c["category"] == "appetizer"), "-"),
            "Main dish": next((c["name"] for c in r["courses"] if c["category"] == "main_dish"), "-"),
            "Dessert": next((c["name"] for c in r["courses"] if c["category"] == "dessert"), "-"),
            "Health": round(r["health_score"], 2),
            "FSA ↓": round(r["fsa"], 2),
            "WHO ↑": round(r["who"], 2),
            "Model score": round(r["score"], 4),
            **({"Preference": round(r["preference"], 4), "Health bonus": round(r["health_bonus"], 4)}
               if r["preference"] is not None else {}),
            "In test": "✅" if r["in_test_set"] else "",
        } for r in rows])
        st.dataframe(table, hide_index=True, width="stretch")

    with st.expander("User history (train interactions)"):
        hist = pd.DataFrame({"meal_id": profile["history"], "health_score": np.round(hist_h, 2)})
        st.dataframe(hist, hide_index=True, width="stretch")


# ---------------- navigation ----------------
pages = [st.Page(page_meals, title="Meals for you", icon=":material/restaurant:", default=True),
         st.Page(page_insights, title="Model insights", icon=":material/insights:")]
nav = st.navigation(pages, position="hidden")


def pick_random_user():
    st.session_state["user_id"] = int(np.random.choice(pool))


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark"></div><div><div class="brand-name">HealthRank</div>'
                '<div class="brand-tag">Familiar meals, a little healthier</div></div></div>', unsafe_allow_html=True)
    for i, pg_ in enumerate(pages):
        with st.container(key=f"nav_{'on' if pg_.title == nav.title else 'off'}_{i}"):
            st.page_link(pg_)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.selectbox("Example user", pool, key="user_id", format_func=lambda u: f"User {u}",
                 help="Anonymised users from the MealRec+ dataset.")
    st.selectbox("Number of meals", [6, 9, 12, 15], key="k")
    st.button("Try a random user", width="stretch", on_click=pick_random_user)
    st.markdown('<div class="side-note">Explore recommendations for an existing user from the MealRec+ dataset.</div>',
                unsafe_allow_html=True)
nav.run()

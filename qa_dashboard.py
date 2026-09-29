"""Burn-in QA Inspector Dashboard (hackathon pitch app).

Modules:
  A. Lot-relative dynamic outlier detector (Modified Z-score on MAD).
     Flags parts anomalous vs their LOT even when under the absolute datasheet limit.
     Story example: 45 uA in a lot averaging 10 uA is anomalous though limit is 50 uA.
  B. Drift predictor: Ridge regression on [0h, 24h] -> 168h forecast.
     Flags early rejection when predicted drift slope exceeds a safety slope.
     Evaluated with heavy (10x) penalty on false negatives.

LLM explanations come from local Ollama (llama3.1), with a rule-based fallback
if Ollama is unreachable.

Run:  streamlit run qa_dashboard.py
"""

from __future__ import annotations

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:latest"
FN_COST = 10  # a missed bad part costs 10x a false alarm
HOURS = [0, 24, 96, 168]

st.set_page_config(page_title="Burn-In QA Dashboard", layout="wide")

# ---------------------------------------------------------------- clean light UI
st.markdown(
    """
<style>
  .stApp { background: #FAFBFC; color: #1A2332; }
  section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E8EE; }
  .kpi { background: #FFFFFF; border: 1px solid #E5E8EE; border-radius: 10px;
         padding: 12px 16px; }
  .kpi .lbl { font-size: 12px; color: #5B6472; text-transform: uppercase; letter-spacing: .04em; }
  .kpi .val { font-size: 24px; font-weight: 700; margin-top: 2px; }
  .card { background: #FFFFFF; border: 1px solid #E5E8EE; border-radius: 10px; padding: 16px 18px; }
  .badge { display: inline-block; font-size: 12px; font-weight: 700; border-radius: 999px;
            padding: 2px 10px; }
  .b-red { background: #FDECEC; color: #B91C1C; border: 1px solid #F5C2C2; }
  .b-amber { background: #FFF7E6; color: #B45309; border: 1px solid #F0D9A8; }
  .b-green { background: #EAF7EF; color: #15803D; border: 1px solid #BFE3CB; }
  .b-blue { background: #EAF1FE; color: #1B5BD7; border: 1px solid #C4D6FA; }
  .b-gray { background: #F1F3F6; color: #5B6472; border: 1px solid #E0E4EA; }
  h1, h2, h3 { color: #1A2332 !important; }
  /* sidebar contrast guard: keep labels/radio readable on the white sidebar */
  section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div[role="radiogroup"] {
    color: #1A2332 !important;
  }
  /* light header bar even if a stale dark theme is active */
  [data-testid="stHeader"] { background: #FAFBFC !important; }
  /* expander (pitch script): explicit light card so text never goes white-on-white */
  [data-testid="stExpander"] { background: #FFFFFF !important; border: 1px solid #E5E8EE;
    border-radius: 10px; }
  [data-testid="stExpander"] summary *, [data-testid="stExpanderDetails"] * { color: #1A2332 !important; }
  /* HTML results table (theme-independent, replaces st.dataframe) */
  table.qa-table { width: 100%; border-collapse: collapse; font-size: 13px; color: #1A2332;
    background: #FFFFFF; }
  table.qa-table th { background: #1A2332; color: #FFFFFF; text-align: left;
    padding: 8px 10px; font-size: 12px; }
  table.qa-table th:first-child { border-radius: 8px 0 0 0; }
  table.qa-table th:last-child { border-radius: 0 8px 0 0; }
  table.qa-table td { border-top: 1px solid #E5E8EE; padding: 7px 10px; }
  table.qa-table tr:nth-child(even) td { background: #F5F7FA; }
  table.qa-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .flag-yes { color: #B91C1C; font-weight: 700; }
  .flag-no { color: #9AA4B2; }
  .stButton>button { border-radius: 8px; }
  #MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ synthetic burn-in data
LOTS = {"LOT-2417": 10.0, "LOT-2418": 12.0, "LOT-2419": 9.0}


@st.cache_data(show_spinner=False)
def generate_burnin(seed: int = 7, n_per_lot: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lot, base in LOTS.items():
        for i in range(n_per_lot):
            pid = f"{lot}-P-{i:03d}"
            v0 = base + rng.normal(0, 1.2)
            drift = rng.normal(0.6, 0.9)  # healthy total drift over 168h
            v24 = v0 + drift * 24 / 168 + rng.normal(0, 0.4)
            v96 = v0 + drift * 96 / 168 + rng.normal(0, 0.5)
            v168 = v0 + drift + rng.normal(0, 0.5)
            rows.append([pid, lot, v0, v24, v96, v168])
    df = pd.DataFrame(rows, columns=["part_id", "lot_id", "v0", "v24", "v96", "v168"])
    # --- injected pitch stories (deterministic, one set per lot) ---
    stories = [
        ("STORY", 11.0, 14.0, 28.0, 45.0),  # passes 50 limit, anomalous vs ~10 lot
        ("DRIFT", 10.5, 18.0, 35.0, 58.0),  # fast drifter, fails limit + slope
        ("OK", 10.2, 10.8, 11.1, 11.4),  # healthy reference
    ]
    extra = [[f"{lot}-P-{tag}", lot, v0, v24, v96, v168]
             for lot in LOTS for tag, v0, v24, v96, v168 in stories]
    df = pd.concat([df, pd.DataFrame(extra, columns=df.columns)], ignore_index=True)
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_drift_model(_df: pd.DataFrame):
    X = _df[["v0", "v24"]].to_numpy()
    y = _df["v168"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(Xtr, ytr)
    mae = mean_absolute_error(yte, model.predict(Xte))
    return model, mae


def modified_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = float(np.median(np.abs(s - med)))
    if mad == 0:
        return pd.Series(0.0, index=s.index)
    return 0.6745 * (s - med) / mad


def drift_slope(v_start: float, v_end: float, h_start: int = 24, h_end: int = 168) -> float:
    return (v_end - v_start) / (h_end - h_start)


def verdict_badge(text: str, kind: str) -> str:
    return f'<span class="badge b-{kind}">{text}</span>'


def style_fig(fig: go.Figure, title: str) -> go.Figure:
    """Explicit light styling so charts read the same in any Streamlit theme."""
    fig.update_layout(
        font=dict(color="#1A2332", size=12),
        title=dict(text=title, font=dict(color="#1A2332", size=14)),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=300,
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(font=dict(color="#1A2332")),
    )
    fig.update_xaxes(tickfont=dict(color="#3A4454"), title_font=dict(color="#1A2332"),
                     gridcolor="#E8ECF1", linecolor="#9AA4B2")
    fig.update_yaxes(tickfont=dict(color="#3A4454"), title_font=dict(color="#1A2332"),
                     gridcolor="#E8ECF1", linecolor="#9AA4B2")
    return fig


def rule_fallback(p: dict) -> str:
    lines = [
        f"**Finding:** Part {p['part_id']} reads {p['reading']:.1f} uA at "
        f"{p['checkpoint']}h — {'OVER' if p['over_limit'] else 'under'} the "
        f"{p['limit']:.0f} uA datasheet limit, but its lot-relative score "
        f"(|modified-Z| = {p['z']:.1f}) "
        f"{'exceeds' if p['a_flag'] else 'is within'} the lot threshold {p['z_thr']:.1f}.",
        f"**Why it matters:** The lot median is {p['lot_median']:.1f} uA "
        f"(MAD {p['lot_mad']:.2f}), so this part behaves nothing like its peers. "
        f"Its predicted 168h value is {p['pred168']:.1f} uA "
        f"(drift {p['slope']:.3f} uA/h vs safety {p['safety']:.3f} uA/h) — "
        f"{'early-reject risk' if p['b_flag'] else 'drift looks normal'}.",
        "**Disposition:** QUARANTINE for burn-in extension / FA review.",
    ]
    if not p["a_flag"] and not p["b_flag"]:
        lines[2] = "**Disposition:** ACCEPT — routine flow, no hold."
    elif p["a_flag"] and not p["b_flag"]:
        lines[2] = ("**Disposition:** HOLD for lot-relative review — confirm probe contact, "
                    "re-measure, compare against lot siblings before release.")
    elif p["b_flag"] and not p["a_flag"]:
        lines[2] = ("**Disposition:** EARLY REJECT — predicted drift violates the safety slope; "
                    "pull before 168h burn-in completes.")
    return "\n\n".join(lines)


def llm_explain(payload: dict) -> tuple[str, bool]:
    """Ask local Ollama llama3.1 for a QA explanation. Returns (text, used_llm)."""
    prompt = (
        "You are a semiconductor QA assistant writing for a factory inspector.\n"
        f"Part {payload['part_id']} in {payload['lot_id']}: "
        f"reading {payload['reading']:.1f} uA at {payload['checkpoint']}h; "
        f"datasheet limit {payload['limit']:.0f} uA "
        f"({'OVER' if payload['over_limit'] else 'under'} limit); "
        f"lot median {payload['lot_median']:.1f} uA, MAD {payload['lot_mad']:.2f}, "
        f"|modified-Z| {payload['z']:.1f} vs threshold {payload['z_thr']:.1f} "
        f"({'LOT ANOMALY' if payload['a_flag'] else 'lot-normal'}); "
        f"0h {payload['v0']:.1f}, 24h {payload['v24']:.1f}, "
        f"predicted 168h {payload['pred168']:.1f} uA, "
        f"drift slope {payload['slope']:.3f} uA/h vs safety {payload['safety']:.3f} "
        f"({'DRIFT RISK' if payload['b_flag'] else 'drift OK'}).\n"
        "Reply in exactly 3 short bullets: **Finding**, **Why it matters**, "
        "**Disposition** (ACCEPT / HOLD / EARLY REJECT plus one concrete next step). "
        "Plain language, under 120 words, use the numbers."
    )
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "num_predict": 220}},
            timeout=90,
        )
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        if not text:
            raise ValueError("empty LLM response")
        return text, True
    except Exception:
        return rule_fallback(payload), False


def ollama_ok() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.ok
    except Exception:
        return False


# ------------------------------------------------------------------ sidebar inputs
# NOTE: seed widget comes first so data + model below always match the chosen seed.
with st.sidebar:
    st.header("Test setup")
    seed = st.number_input("Demo seed", 0, 999, 7, 1)
    lot_id = st.selectbox("Lot", sorted(LOTS), index=0)
    checkpoint = st.selectbox("Outlier checkpoint", ["v168", "v96", "v24", "v0"], index=0,
                              format_func=lambda c: c[1:] + "h")
    datasheet_limit = st.number_input("Datasheet limit (uA)", 10.0, 100.0, 50.0, 1.0)
    z_thr = st.slider("Lot-anomaly threshold |Z|", 2.0, 5.0, 3.5, 0.1)
    safety_slope = st.number_input("Safety drift slope (uA/h)", 0.01, 0.50, 0.08, 0.01,
                                   help="Reject if predicted (168h-24h)/144h exceeds this.")
    st.divider()
    st.subheader("Pitch scenarios")
    scen = st.radio("Jump to part", ["Hidden outlier (45 uA story)", "Fast drifter",
                                     "Healthy reference"], index=0)
    default_part = {"Hidden outlier (45 uA story)": f"{lot_id}-P-STORY",
                    "Fast drifter": f"{lot_id}-P-DRIFT",
                    "Healthy reference": f"{lot_id}-P-OK"}[scen]
    with st.expander("60-second pitch script"):
        st.markdown(
            "1. **Hidden outlier:** STORY part reads 45 uA — passes the 50 uA limit, "
            "but |Z| ≈ 10 vs a ~10 uA lot. Module A catches what the datasheet misses.\n"
            "2. **Fast drifter:** DRIFT part is only 18 uA at 24h — looks fine — but the "
            "forecast hits ~58 uA at 168h. Module B early-rejects on slope.\n"
            "3. **Explanation:** click *Explain with LLM* — local llama3.1 writes the "
            "inspector disposition. Misses cost 10x false alarms."
        )

df = generate_burnin(seed=int(seed))
model, drift_mae = train_drift_model(df)

lot = df[df["lot_id"] == lot_id].copy()
lot["z"] = modified_z(lot[checkpoint]).abs()
lot["over_limit"] = lot[checkpoint] > datasheet_limit
lot["a_flag"] = lot["z"] > z_thr
pred168_all = model.predict(lot[["v0", "v24"]].to_numpy())
lot["pred168"] = pred168_all
lot["pred_slope"] = (lot["pred168"] - lot["v24"]) / (168 - 24)
lot["b_flag"] = (lot["pred168"] > datasheet_limit) | (lot["pred_slope"] > safety_slope)
lot["verdict"] = np.select(
    [lot["a_flag"] & lot["b_flag"], lot["a_flag"], lot["b_flag"]],
    ["REJECT — both modules", "HOLD — lot anomaly", "REJECT — drift risk"], "ACCEPT")

# FN-weighted evaluation on this lot (actual = measured 168h truth)
actual_bad = (lot["v168"] > datasheet_limit) | (
    (lot["v168"] - lot["v24"]) / 144 > safety_slope)
pred_bad = lot["b_flag"] | lot["a_flag"]
fn = int(((~pred_bad) & actual_bad).sum())
fp = int((pred_bad & (~actual_bad)).sum())
cost = FN_COST * fn + fp

# ---------------------------------------------------------------------- header
llm_live = ollama_ok()
st.title("Burn-In QA — Lot Inspector")
st.caption(f"Lot-relative outlier detection + 168h drift forecast · "
           f"{'🟢 llama3.1 local (ollama) connected' if llm_live else '🟡 LLM offline — fallback text will be used'}")
c1, c2, c3, c4 = st.columns(4)
lot_med, lot_mad = float(lot[checkpoint].median()), float(np.median(np.abs(lot[checkpoint] - lot[checkpoint].median())))
c1.markdown(f'<div class="kpi"><div class="lbl">Lot {lot_id} · {checkpoint[1:]}h median</div>'
            f'<div class="val">{lot_med:.1f} uA</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi"><div class="lbl">Module A flags (lot anomaly)</div>'
            f'<div class="val">{int(lot["a_flag"].sum())} parts</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi"><div class="lbl">Module B flags (drift risk)</div>'
            f'<div class="val">{int(lot["b_flag"].sum())} parts</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi"><div class="lbl">Miss-weighted cost (FN×{FN_COST}+FP)</div>'
            f'<div class="val">{cost} <span style="font-size:13px;color:#5B6472">({fn} miss · {fp} false alarm)</span></div></div>',
            unsafe_allow_html=True)

# ------------------------------------------------------- Module A + Module B row
colA, colB = st.columns(2)

with colA:
    st.subheader("A · Dynamic outlier detector")
    st.caption(f"Modified-Z = 0.6745·(x − lot median)/MAD on {checkpoint[1:]}h · "
               f"limit {datasheet_limit:.0f} uA is shown, not trusted alone.")
    story_hit = lot[lot["part_id"] == f"{lot_id}-P-STORY"]
    story = story_hit.iloc[0] if len(story_hit) else lot.sort_values("z", ascending=False).iloc[0]
    st.markdown(
        f'<div class="card">Story check — <b>{story["part_id"]}</b>: reads '
        f'<b>{story[checkpoint]:.1f} uA</b> '
        f'({"OVER" if story["over_limit"] else "UNDER"} {datasheet_limit:.0f} uA limit) · '
        f'|Z| = <b>{story["z"]:.1f}</b> vs {z_thr:.1f} → '
        f'{verdict_badge("ANOMALOUS vs lot", "red") if story["a_flag"] else verdict_badge("lot-normal", "green")}'
        f'</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Box(y=lot[checkpoint], name="lot distribution", boxpoints=False,
                         marker_color="#9AA4B2", line_color="#9AA4B2"))
    flagged = lot[lot["a_flag"]]
    fig.add_trace(go.Scatter(y=flagged[checkpoint], x=["lot distribution"] * len(flagged),
                             mode="markers", name="flagged",
                             marker=dict(color="#B91C1C", size=9, symbol="x"),
                             text=flagged["part_id"], hovertemplate="%{text}<br>%{y:.1f} uA"))
    fig.add_hline(y=datasheet_limit, line_dash="dash", line_color="#1B5BD7",
                  annotation_text=f"datasheet {datasheet_limit:.0f} uA",
                  annotation_font_color="#1B5BD7")
    fig.add_hline(y=lot_med, line_color="#5B6472",
                  annotation_text=f"lot median {lot_med:.1f} uA",
                  annotation_font_color="#5B6472")
    style_fig(fig, f"{lot_id} readings at {checkpoint[1:]}h")
    st.plotly_chart(fig, use_container_width=True)
    top = lot.sort_values("z", ascending=False).head(8)
    rows_html = "".join(
        "<tr><td>{pid}</td><td class='num'>{v:.1f}</td><td class='num'>{z:.1f}</td>"
        "<td>{ol}</td><td>{af}</td></tr>".format(
            pid=r["part_id"], v=r[checkpoint], z=r["z"],
            ol="<span class='flag-yes'>OVER</span>" if r["over_limit"] else "<span class='flag-no'>—</span>",
            af="<span class='flag-yes'>FLAG</span>" if r["a_flag"] else "<span class='flag-no'>—</span>",
        )
        for _, r in top.iterrows()
    )
    st.markdown(
        f"<table class='qa-table'><thead><tr><th>part</th><th>{checkpoint[1:]}h (uA)</th>"
        f"<th>|Z| vs lot</th><th>over limit?</th><th>lot anomaly?</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )

with colB:
    st.subheader("B · 168h drift predictor")
    st.caption(f"Ridge on [0h, 24h] → 168h · test MAE {drift_mae:.2f} uA · "
               f"safety slope {safety_slope:.2f} uA/h · misses cost {FN_COST}×.")
    st.markdown('<div class="card"><b>New-part early check</b> — enter 0h/24h, forecast 168h.</div>',
                unsafe_allow_html=True)
    n0 = st.number_input("0-hour reading (uA)", 0.0, 80.0, 11.0, 0.5)
    n24 = st.number_input("24-hour reading (uA)", 0.0, 80.0, 14.0, 0.5)
    p168 = float(model.predict(np.array([[n0, n24]]))[0])
    slope = drift_slope(n24, p168)
    b_risk = (p168 > datasheet_limit) or (slope > safety_slope)
    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="kpi"><div class="lbl">Predicted 168h</div>'
                f'<div class="val">{p168:.1f} uA</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="kpi"><div class="lbl">Drift slope</div>'
                f'<div class="val">{slope:.3f} uA/h</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="kpi"><div class="lbl">Verdict</div>'
                f'<div class="val" style="font-size:15px;padding-top:4px">'
                f'{verdict_badge("EARLY REJECT", "red") if b_risk else verdict_badge("OK so far", "green")}'
                f'</div></div>', unsafe_allow_html=True)
    traj = lot.sort_values("pred_slope", ascending=False).head(4)
    fig2 = go.Figure()
    for _, r in traj.iterrows():
        fig2.add_trace(go.Scatter(x=HOURS, y=[r["v0"], r["v24"], r["v96"], r["v168"]],
                                  mode="lines+markers", name=r["part_id"].split("-")[-1],
                                  line=dict(width=2),
                                  marker=dict(size=6)))
    fig2.add_trace(go.Scatter(x=[24, 168], y=[n24, p168], mode="lines+markers",
                              name="your input → forecast",
                              line=dict(dash="dash", color="#1B5BD7", width=3),
                              marker=dict(size=8)))
    fig2.add_hline(y=datasheet_limit, line_dash="dot", line_color="#B91C1C",
                   annotation_text="limit", annotation_font_color="#B91C1C")
    style_fig(fig2, "Burn-in trajectories (top drifters + your forecast)")
    fig2.update_layout(xaxis_title="hours", yaxis_title="uA")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------- part inspector
st.subheader("Part inspector — inspector-ready explanation")
parts = lot.sort_values(["a_flag", "b_flag", "z"], ascending=[False, False, False])["part_id"].tolist()
sel = st.selectbox("Select part", parts,
                   index=parts.index(default_part) if default_part in parts else 0)
r = lot[lot["part_id"] == sel].iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="kpi"><div class="lbl">Reading @ {checkpoint[1:]}h</div>'
            f'<div class="val">{r[checkpoint]:.1f} uA</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi"><div class="lbl">Lot-relative |Z|</div>'
            f'<div class="val">{r["z"]:.1f}</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi"><div class="lbl">Predicted 168h</div>'
            f'<div class="val">{r["pred168"]:.1f} uA</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi"><div class="lbl">Disposition</div>'
            f'<div class="val" style="font-size:16px">{r["verdict"]}</div></div>', unsafe_allow_html=True)

payload = dict(part_id=r["part_id"], lot_id=lot_id, reading=float(r[checkpoint]),
               checkpoint=checkpoint[1:] + "h", limit=float(datasheet_limit),
               over_limit=bool(r["over_limit"]), z=float(r["z"]), z_thr=float(z_thr),
               a_flag=bool(r["a_flag"]), lot_median=lot_med, lot_mad=lot_mad,
               v0=float(r["v0"]), v24=float(r["v24"]),
               pred168=float(r["pred168"]), slope=float(r["pred_slope"]),
               safety=float(safety_slope), b_flag=bool(r["b_flag"]))

if st.button("Explain with LLM (local llama3.1)", type="primary"):
    with st.spinner("llama3.1 is writing the inspector note…"):
        text, used = llm_explain(payload)
    st.markdown(f'<div class="card">{text}'
                f'<div style="margin-top:8px;font-size:12px;color:#5B6472">'
                f'{"Generated locally by llama3.1 via Ollama" if used else "Ollama unreachable — rule-based fallback shown"}. '
                f'Misses cost {FN_COST}× false alarms; when in doubt the model holds the part.</div></div>',
                unsafe_allow_html=True)

st.caption("Method: A = |0.6745·(x−median)/MAD| per lot; B = Ridge[0h,24h]→168h, "
           "reject if forecast > limit or slope > safety. Thresholds favour recall: false negatives penalised 10×.")

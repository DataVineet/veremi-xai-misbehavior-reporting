"""Streamlit demonstration of the full pipeline (presentation layer only — all logic lives in src/veremi_xai).

Run from the project root:   .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from veremi_xai.config import RAW_FEATURES, load_config, path
from veremi_xai.explain import OUTPUT_SPACE
from veremi_xai.features import DESCRIPTIONS
from veremi_xai.pipeline import Pipeline
from veremi_xai.reporting import available_backends, generate_report
from veremi_xai.viz import BLUE, CATEGORY_COLORS, ORANGE, POLARITY

st.set_page_config(page_title="C-ITS Misbehavior Reporting", page_icon="🚦", layout="wide")
CFG = load_config()
CLASSES = CFG["classes"]
NAME = {k: v["name"] for k, v in CLASSES.items()}
EDITABLE = ["posx", "posy", "spdx", "spdy", "aclx", "acly", "hedx", "hedy"]


@st.cache_resource(show_spinner="Loading detector and SHAP explainer…")
def get_pipeline() -> Pipeline:
    return Pipeline.load()


@st.cache_data(show_spinner="Loading demo messages…")
def get_demo() -> pd.DataFrame:
    return pd.read_parquet(path("final") / "demo_messages.parquet")


@st.cache_data
def get_experiments() -> pd.DataFrame:
    rows = []
    for p in sorted((path("results") / "experiments").glob("*.json")):
        r = json.loads(p.read_text())
        rows.append({"experiment": r["experiment"], "model": r["model"], "features": f'{r["features"]} ({r["n_features"]})', "identity": r["identity"],
                     "split": r["split"], "weighted": r["class_weighted"], **{k: r["test"][k] for k in ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1", "mcc", "binary_f1_misbehaving", "sender_macro_f1")},
                     "fit_s": r["fit_seconds"], "ms_per_1k_msgs": r["test"]["predict_ms_per_1k"]})
    return pd.DataFrame(rows)


def label(c: int) -> str:
    return f"{c} · {NAME[c]}"


pipe, demo = get_pipeline(), get_demo()
card = pipe.card

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🚦 Misbehavior reporting")
    st.caption("LLM-assisted explainable misbehavior reporting framework for C-ITS · VeReMi Extension")
    st.markdown(f"**Detector** {card['algorithm'].split(',')[0]}  \n**Features** {card['n_features']} ({card['feature_set']})  \n"
                f"**Test macro-F1** {card['test_macro_f1']} · **accuracy** {card['test_accuracy']}  \n"
                f"<small>sender-disjoint test set, 20 classes</small>", unsafe_allow_html=True)
    st.divider()
    backends = available_backends()
    opts = [b for b, ok in backends.items() if ok]
    preferred = CFG["llm"].get("app_default")
    default = preferred if preferred in opts else next((b for b in opts if b not in ("template", "ollama")), "template")
    backend = st.selectbox("Report writer", opts, index=opts.index(default),
                           help="`template` is a deterministic offline writer. LLM profiles appear when their API key is set in `.env` "
                                "(see `.env.example`). Every LLM report is validated against the evidence; on failure the template is used.")
    missing = [b for b, ok in backends.items() if not ok]
    if missing:
        st.caption("No API key found for: " + ", ".join(missing))
    top_k = st.slider("SHAP features in the evidence package", 3, 10, 6)

tab_detect, tab_perf, tab_global, tab_about = st.tabs(["Detect & explain", "Model performance", "Global explainability", "About & limitations"])

# ------------------------------------------------------------------ 1. detect & explain
with tab_detect:
    st.subheader("1 · Select a vehicle message")
    st.caption("All messages below come from **test vehicles the model never saw during training**.")
    showcase_file = path("final") / "showcase_messages.json"
    showcase = json.loads(showcase_file.read_text(encoding="utf-8")) if showcase_file.exists() else []
    quick = st.selectbox("Quick example (curated test messages, good for a first look)", ["Browse manually"] + [e["label"] for e in showcase])
    if quick != "Browse manually":
        chosen = next(e for e in showcase if e["label"] == quick)
        veh = demo[demo.sender == chosen["sender"]].sort_values("sendTime").reset_index(drop=True)
        pos = int(veh.index[veh.messageID == chosen["messageID"]][0])
        msg = veh.loc[pos]
    else:
        picks = demo[demo.in_demo_pick]
        c1, c2, c3 = st.columns([2, 2, 3])
        cls = c1.selectbox("True class of the vehicle (ground truth, used only for browsing)", list(NAME), format_func=label, index=13)
        outcome = c2.radio("Model outcome", ["Any", "Correct", "Misclassified"], horizontal=True)
        pool = picks[picks["class"] == cls]
        if outcome != "Any":
            pool = pool[(pool.pred == pool["class"]) == (outcome == "Correct")]
        if pool.empty:
            st.info("No demo message matches this filter — choose another outcome or class.")
            st.stop()
        senders = pool.sender.unique()
        sender = c3.selectbox("Vehicle (sender id — simulator ground truth, never shown to the model)", senders)
        veh = demo[demo.sender == sender].sort_values("sendTime").reset_index(drop=True)
        cand = veh[veh.messageID.isin(pool.messageID)]
        pos = st.select_slider("Message (by send time, s)", options=list(cand.index), value=int(cand.index[len(cand) // 2]),
                               format_func=lambda i: f"{veh.sendTime[i]:.2f}")
        msg = veh.loc[pos]

    ok = np.where(veh.pred == veh["class"], "prediction = label", "prediction ≠ label")
    tl = veh.assign(outcome=ok, speed=np.hypot(veh.spdx, veh.spdy), predicted=veh.pred.map(NAME))
    base = alt.Chart(tl).encode(x=alt.X("sendTime:Q", title="send time (s)", scale=alt.Scale(zero=False)))
    pts = base.mark_circle(size=42, opacity=0.85).encode(
        y=alt.Y("speed:Q", title="transmitted speed (m/s)"),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["prediction = label", "prediction ≠ label"], range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("outcome:N", scale=alt.Scale(domain=["prediction = label", "prediction ≠ label"], range=["circle", "cross"]), legend=None),
        tooltip=["messageID", alt.Tooltip("sendTime:Q", format=".2f"), alt.Tooltip("speed:Q", format=".2f"), "senderPseudo", "predicted", alt.Tooltip("conf:Q", format=".3f", title="confidence")])
    rule = alt.Chart(pd.DataFrame({"sendTime": [msg.sendTime]})).mark_rule(color="#52514e", strokeDash=[4, 3]).encode(x="sendTime:Q")
    st.altair_chart((pts + rule).properties(height=190, title=f"All {len(veh)} messages of this vehicle — dashed line = selected message"), width="stretch")

    with st.expander("What-if: edit the transmitted values of this message before scoring"):
        st.caption("Edits apply to this message only; history-based features are recomputed by the same feature code used in training.")
        cols = st.columns(4)
        edited = {f: cols[i % 4].number_input(f, value=float(msg[f]), format="%.4f", key=f"{msg.messageID}_{f}") for i, f in enumerate(EDITABLE)}
    # history = every demo message sent under the same pseudonym (what a receiver could have observed)
    work = demo[demo.senderPseudo == msg.senderPseudo].copy()
    changed = any(abs(edited[f] - float(msg[f])) > 1e-9 for f in EDITABLE)
    for f in EDITABLE:
        work.loc[work.messageID == msg.messageID, f] = edited[f]

    res = pipe.analyse(work[["sendTime", "sender", "senderPseudo", "messageID", "class"] + RAW_FEATURES], int(msg.messageID), top_k=top_k)
    ev, proba = res["evidence"], res["proba"]
    P = ev["prediction"]

    st.subheader("2 · ML prediction and confidence")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted class", P["class_name"])
    m2.metric("Category", P["category"])
    m3.metric("Confidence", f"{P['confidence']:.1%}", P["confidence_level"], delta_color="off")
    truth = NAME[int(msg["class"])]
    m4.metric("Ground truth (not given to model or LLM)", truth, "edited message" if changed else ("match" if truth == P["class_name"] else "mismatch"), delta_color="off")
    st.caption(f"**Definition** — {P['definition']}")

    left, right = st.columns([2, 3])
    with left:
        top5 = pd.DataFrame({"class": [NAME[i] for i in np.argsort(proba)[::-1][:5]], "probability": np.sort(proba)[::-1][:5]})
        ch = alt.Chart(top5).mark_bar(color=BLUE, cornerRadiusEnd=4, height=16).encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")), y=alt.Y("class:N", sort=None, title=None),
            tooltip=["class", alt.Tooltip("probability:Q", format=".4f")])
        txt = ch.mark_text(align="left", dx=4, color="#52514e").encode(text=alt.Text("probability:Q", format=".1%"))
        st.altair_chart((ch + txt).properties(height=190, title="Top-5 class probabilities"), width="stretch")
    with right:
        sh = pd.DataFrame(ev["shap"]["top_contributions"])
        sh["label"] = sh.apply(lambda r: f"{r.feature} = {'n/a' if r.value is None or pd.isna(r.value) else r.value}", axis=1)
        ch = alt.Chart(sh).mark_bar(cornerRadiusEnd=4, height=16).encode(
            x=alt.X("shap:Q", title=f"SHAP contribution to '{P['class_name']}' (raw margin)"), y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("effect:N", scale=alt.Scale(domain=list(POLARITY), range=list(POLARITY.values())), legend=alt.Legend(title=None, orient="top")),
            tooltip=["feature", "value", "shap", "effect", "description"])
        st.altair_chart(ch.properties(height=38 + 28 * len(sh), title="3 · SHAP: why the model predicted this class"), width="stretch")
    st.caption(f"SHAP values are exact TreeSHAP contributions in the {OUTPUT_SPACE}: base value {ev['shap']['base_value']} + all contributions = model output "
               f"{ev['shap']['model_output']}. The {len(sh)} features shown carry {ev['shap']['share_of_total_attribution_in_top']:.0%} of the total absolute attribution. "
               "They explain the model, not the physical world.")

    with st.expander("Feature values and meanings for this message"):
        tbl = pd.DataFrame({"feature": pipe.features, "value": [res["row"][f] for f in pipe.features],
                            "SHAP (predicted class)": res["shap_row"][:, P["class_id"]], "meaning": [DESCRIPTIONS[f] for f in pipe.features]})
        st.dataframe(tbl.sort_values("SHAP (predicted class)", key=np.abs, ascending=False), hide_index=True, width="stretch")

    st.subheader("4 · Structured evidence package")
    st.caption("This JSON is the **only** input of the report writer. It contains no ground truth and nothing from the rest of the dataset.")
    with st.expander("Show evidence JSON"):
        st.json(ev)
    st.download_button("Download evidence JSON", json.dumps(ev, indent=1), file_name=f"evidence_{ev['message']['messageID']}.json")

    st.subheader("5 · Human-readable misbehavior report")
    if st.button(f"Generate report with `{backend}`", type="primary"):
        with st.spinner("Writing and validating report…"):
            st.session_state["report"] = (ev["evidence_hash"], generate_report(ev, backend=backend))
    rep = st.session_state.get("report")
    if rep and rep[0] == ev["evidence_hash"]:
        rep = rep[1]
        if rep.fallback_reason:
            st.warning(f"Fell back to the template writer — {rep.fallback_reason}")
        with st.container(border=True):
            st.markdown(rep.text)
        v = rep.validation
        (st.success if v["passed"] else st.error)(
            ("✅ Faithfulness validation passed" if v["passed"] else "❌ Faithfulness validation failed") +
            f" · {v['numbers_checked']} numbers checked against the evidence · " + " · ".join(f"{k}: {'ok' if ok else 'FAIL'}" for k, ok in v["checks"].items()))
        st.caption(f"writer: {rep.backend} · model: {rep.model} · prompt {rep.prompt_version} · evidence {rep.evidence_hash} · {rep.created_utc}" + (" · from cache" if rep.from_cache else ""))
        if rep.rejected_llm_text:
            with st.expander("Rejected LLM draft (failed validation)"):
                st.markdown(rep.rejected_llm_text)
        st.download_button("Download report (Markdown)", rep.text, file_name=f"report_{ev['message']['messageID']}.md")
    else:
        st.info("Press the button to turn the evidence package into a report.")

# ------------------------------------------------------------------ 2. performance
with tab_perf:
    st.subheader("Final detector on the sender-disjoint test set")
    t = card["test"]
    cols = st.columns(6)
    for col, (k, lab) in zip(cols, [("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced acc."), ("accuracy", "Accuracy"), ("mcc", "MCC"),
                                    ("binary_f1_misbehaving", "Binary F1 (misbehaving)"), ("sender_macro_f1", "Vehicle-level macro-F1")]):
        col.metric(lab, f"{t[k]:.3f}")
    st.caption("Test vehicles are disjoint from training vehicles. Accuracy is never reported alone because 59.5 % of messages are genuine.")
    final = json.loads((path("results") / "experiments" / f"{card['experiment']}.json").read_text())
    pc = pd.DataFrame(final["test_per_class"]); pc["category"] = pc["class"].map(lambda c: CLASSES[c]["group"])
    bar = alt.Chart(pc).mark_bar(cornerRadiusEnd=4, height=13).encode(
        x=alt.X("f1:Q", scale=alt.Scale(domain=[0, 1]), title="F1 on test set"), y=alt.Y("name:N", sort=None, title=None, axis=alt.Axis(labelLimit=200)),
        color=alt.Color("category:N", scale=alt.Scale(domain=list(CATEGORY_COLORS), range=list(CATEGORY_COLORS.values())), legend=alt.Legend(orient="top", title=None)),
        tooltip=["class", "name", "category", alt.Tooltip("precision:Q", format=".3f"), alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("f1:Q", format=".3f"), "support"])
    a, b = st.columns([3, 2])
    a.altair_chart(bar.properties(height=470, title="Per-class F1"), width="stretch")
    b.dataframe(pc[["class", "name", "precision", "recall", "f1", "support"]], hide_index=True, width="stretch", height=500)
    fig = path("figures") / "final_confusion_matrix.png"
    if fig.exists():
        st.image(str(fig), caption="Row-normalised confusion matrix (test set)")
    st.subheader("All experiments (test set)")
    st.caption("`split = row` is the naive random row split used in most of the literature: the same vehicle appears in train and test, so the score is inflated. "
               "`identity` is the key used for history features: `senderPseudo` is what a receiver can observe; `sender` is simulator ground truth.")
    st.dataframe(get_experiments().sort_values("macro_f1", ascending=False).style.format(precision=4), hide_index=True, width="stretch")
    for f, cap in (("experiment_comparison.png", "Feature-set ablation, model comparison and leakage demonstration"),):
        if (path("figures") / f).exists():
            st.image(str(path("figures") / f), caption=cap)

# ------------------------------------------------------------------ 3. global explainability
with tab_global:
    st.subheader("What the detector relies on overall")
    gi = path("results") / "shap_global_importance.csv"
    if gi.exists():
        g = pd.read_csv(gi, index_col=0).reset_index().rename(columns={"index": "feature"}).head(20)
        g["meaning"] = g.feature.map(DESCRIPTIONS)
        ch = alt.Chart(g).mark_bar(color=BLUE, cornerRadiusEnd=4, height=13).encode(
            x=alt.X("mean_abs_shap:Q", title="mean |SHAP| over all classes (raw margin)"), y=alt.Y("feature:N", sort=None, title=None),
            tooltip=["feature", alt.Tooltip("mean_abs_shap:Q", format=".3f"), "meaning"])
        st.altair_chart(ch.properties(height=460, title="Top-20 features by mean |SHAP| (3,000 test messages, 150 per class)"), width="stretch")
    fig = path("figures") / "shap_per_class_heatmap.png"
    if fig.exists():
        st.image(str(fig), caption="Which features drive each class (mean |SHAP| of the class's own score over messages of that class)")
    sc = path("results") / "shap_sanity_checks.json"
    if sc.exists():
        s = json.loads(sc.read_text())
        st.subheader("Are the explanations faithful to the model?")
        k1, k2, k3 = st.columns(3)
        k1.metric("Additivity error (max)", f"{s['additivity_max_abs_error']:.1e}", help="base value + Σ SHAP must equal the model output")
        k2.metric("Prob. drop — top-3 SHAP features replaced", f"{s['perturbation_all']['top3_shap']['mean_prob_drop']:.3f}")
        k3.metric("Prob. drop — 3 random features replaced", f"{s['perturbation_all']['random3']['mean_prob_drop']:.3f}")
        st.caption(s["note"] + ". A much larger drop for the SHAP-selected features shows the explanations point at what the model actually uses.")
    rs = path("results") / "report_faithfulness_summary.csv"
    if rs.exists():
        st.subheader("Report faithfulness (automatic validation)")
        st.dataframe(pd.read_csv(rs), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 4. about
with tab_about:
    st.subheader("Pipeline")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style="rounded", fontname="Helvetica", fontsize=11];
        A [label="VeReMi Extension\\nmessages"]; B [label="Feature preparation\\n(per-message + causal history\\n+ road-map plausibility)"];
        C [label="ML detector\\n(XGBoost, 20 classes)"]; D [label="Prediction +\\nconfidence"]; E [label="SHAP\\n(exact TreeSHAP)"];
        F [label="Structured\\nevidence package"]; G [label="Report writer\\n(LLM or template)"]; H [label="Faithfulness\\nvalidator"]; I [label="Human-readable\\nreport"];
        A->B->C->D->F; C->E->F; F->G->H->I; H->G [label="retry / fallback", fontsize=9]; }""")
    st.markdown("""
**Roles.** The ML model is the only detector. SHAP explains that model's score. The LLM only rewrites the evidence package into prose —
it cannot change the prediction, and every number, class and feature it mentions is checked against the evidence. If the LLM is unavailable
or fails validation twice, a deterministic template report is used instead.
""")
    st.subheader("Classes")
    st.dataframe(pd.DataFrame([{"id": k, "name": v["name"], "category": v["group"], "definition": v["definition"]} for k, v in CLASSES.items()]), hide_index=True, width="stretch")
    st.subheader("Limitations")
    for x in ev["limitations"] + ["Labels are per vehicle: genuine-looking messages of an attacker (e.g. before an 'eventual stop') count as errors, so per-class recall has a ceiling below 100 %.",
                                  "Replay-type attacks copy genuine messages bit-for-bit; they are only detectable through history and identity behaviour.",
                                  "The validator checks that numbers exist in the evidence, not that each number is attached to the right feature."]:
        st.markdown(f"- {x}")

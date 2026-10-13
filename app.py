"""
Assignment 1 — Gradio dashboard.

Loads the preprocessor and the three models saved by train.py and only runs
inference. Nothing here trains. Run `uv run python main.py train` first.
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_curve

from pipeline import (
    ARTIFACTS, CATEGORICAL, COST_PER_CALL, NUMERIC, VALUE_PER_SUBSCRIPTION,
    LogisticRegressionNet, business_value, chronological_split, engineer,
    load_raw,
)

MODEL_NAMES = {"sklearn": "scikit-learn", "manual_torch": "Manual PyTorch",
               "nn_torch": "nn.Module + optim"}

# --- Load artifacts (inference only) ---------------------------------------
if not (ARTIFACTS / "results.json").exists():
    raise SystemExit("No artifacts found. Run `uv run python main.py train` first.")

pre = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_logreg.joblib")
manual = torch.load(ARTIFACTS / "manual_torch.pt")
nn_model = LogisticRegressionNet(len(pre.feature_names_))
nn_model.load_state_dict(torch.load(ARTIFACTS / "nn_torch.pt"))
nn_model.eval()
with open(ARTIFACTS / "results.json") as f:
    RESULTS = json.load(f)
CFG = RESULTS["config"]

raw = load_raw()
splits = dict(zip(["train", "val", "test"], chronological_split(raw)))


def predict_all(df: pd.DataFrame) -> dict:
    X = pre.transform(df)
    Xt = torch.tensor(X)
    with torch.no_grad():
        return {
            "sklearn": sk_model.predict_proba(X)[:, 1],
            "manual_torch": torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy(),
            "nn_torch": torch.sigmoid(nn_model(Xt)).numpy(),
        }


PREDS = {name: predict_all(d) for name, d in splits.items() if name != "train"}


# --- Tab 1: model comparison -----------------------------------------------
def summary_table(split):
    rows = []
    for key, r in RESULTS["results"][split].items():
        b = r.get("business_at_threshold", r.get("business"))
        rows.append({
            "model": MODEL_NAMES.get(key, key),
            "ROC-AUC": round(r["roc_auc"], 4) if "roc_auc" in r else None,
            "PR-AUC": round(r["pr_auc"], 4) if "pr_auc" in r else None,
            "log loss": round(r["log_loss"], 4) if "log_loss" in r else None,
            "precision top 10%": round(r["precision_top10"], 3) if "precision_top10" in r else None,
            f"calls @ t={CFG['threshold']}": b["calls"],
            "profit (EUR)": round(b["profit"]),
        })
    return pd.DataFrame(rows)


def pred_vs_actual(split, model_key):
    y = splits[split]["y"].to_numpy()
    p = PREDS[split][model_key]
    # Calibration view: bin predicted probability, compare with observed rate.
    bins = pd.qcut(p, q=10, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
    fig = make_calibration(cal, split, model_key)
    fig2 = go.Figure()
    for k, name in MODEL_NAMES.items():
        fpr, tpr, _ = roc_curve(y, PREDS[split][k])
        fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
    fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="baseline",
                              line=dict(dash="dash", color="gray")))
    fig2.update_layout(title=f"ROC curves, {split} set (the three lines overlap)",
                       xaxis_title="False positive rate", yaxis_title="True positive rate")
    return fig, fig2


def make_calibration(cal, split, model_key):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal["predicted"], y=cal["observed"], mode="markers+lines",
                             name="decile of predicted probability",
                             text=[f"n={n}" for n in cal["n"]]))
    hi = float(max(cal["predicted"].max(), cal["observed"].max())) * 1.05
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="perfect calibration",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(title=f"Predicted vs. actual subscription rate — {MODEL_NAMES[model_key]}, {split}",
                      xaxis_title="Mean predicted probability", yaxis_title="Observed rate")
    return fig


def model_scatter(split):
    p = PREDS[split]
    df = pd.DataFrame({"scikit-learn": p["sklearn"], "Manual PyTorch": p["manual_torch"],
                       "nn.Module + optim": p["nn_torch"]})
    fig = px.scatter_matrix(df, title=f"Per-client probabilities, model vs. model ({split})")
    fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.4))
    return fig


def coef_plot():
    names = RESULTS["feature_names"]
    df = pd.DataFrame({MODEL_NAMES[k]: v for k, v in RESULTS["coefficients"].items()}, index=names)
    df = df.drop(index="(intercept)").reindex(
        df.drop(index="(intercept)")["scikit-learn"].abs().sort_values().index)
    long = df.reset_index(names="feature").melt(id_vars="feature", var_name="model", value_name="coef")
    fig = px.bar(long, x="coef", y="feature", color="model", barmode="group", orientation="h",
                 height=900, title="Coefficients (standardised numerics, drop-first dummies)")
    return fig


def loss_plot():
    fig = go.Figure()
    for k in ("manual_torch", "nn_torch"):
        h = RESULTS["loss_history"][k]
        fig.add_trace(go.Scatter(y=h, mode="lines", name=MODEL_NAMES[k]))
    fig.add_hline(y=RESULTS["agreement"]["final_objective"]["sklearn"], line_dash="dash",
                  annotation_text="sklearn optimum")
    fig.update_layout(title="Training objective (BCE + L2) per epoch", yaxis_type="log",
                      xaxis_title="epoch", yaxis_title="objective")
    return fig


# --- Tab 2: distributions ---------------------------------------------------
DIST_FEATURES = ["age", "campaign", "previous", "pdays", "euribor3m", "duration"] + \
    [c for c in CATEGORICAL if c != "age_band"]


def distribution(feature, by_split):
    df = pd.concat([engineer(d).assign(split=n) for n, d in splits.items()])
    df["subscribed"] = df["y"].map({0: "no", 1: "yes"})
    color = "split" if by_split else "subscribed"
    if not pd.api.types.is_numeric_dtype(df[feature]):
        rate = df.groupby([color, feature])["y"].mean().reset_index() if by_split else \
            df.groupby(feature)["y"].mean().reset_index().assign(**{color: "all"})
        fig = px.histogram(df, x=feature, color=color, barmode="group", histnorm="percent",
                           title=f"{feature}: share of rows")
        fig2 = px.bar(rate, x=feature, y="y", color=color, barmode="group",
                      title=f"Subscription rate by {feature}", labels={"y": "subscription rate"})
    else:
        fig = px.histogram(df, x=feature, color=color, barmode="overlay", histnorm="percent",
                           nbins=50, opacity=0.6, title=f"{feature}: distribution")
        fig2 = px.line(df.groupby(df["t"] // 500).agg(t=("t", "min"), rate=("y", "mean"),
                                                      value=(feature, "mean")),
                       x="t", y=["rate", "value"], facet_row="variable",
                       title=f"Over time (blocks of 500 contacts): subscription rate and mean {feature}")
        fig2.update_yaxes(matches=None)
    return fig, fig2


def target_over_time():
    df = raw.assign(block=raw["t"] // 500).groupby("block").agg(
        t=("t", "min"), rate=("y", "mean")).reset_index()
    fig = px.line(df, x="t", y="rate", title="Target drift: subscription rate over contact order")
    i_tr, i_va = len(splits["train"]), len(splits["train"]) + len(splits["val"])
    fig.add_vrect(x0=0, x1=i_tr, fillcolor="blue", opacity=0.07, annotation_text="train")
    fig.add_vrect(x0=i_tr, x1=i_va, fillcolor="orange", opacity=0.1, annotation_text="val")
    fig.add_vrect(x0=i_va, x1=len(raw), fillcolor="green", opacity=0.07, annotation_text="test")
    return fig


# --- Tab 3: threshold & business cost --------------------------------------
def threshold_view(threshold, split, model_key, cost, value):
    y = splits[split]["y"].to_numpy()
    p = PREDS[split][model_key]
    call = p >= threshold
    tp = int(np.sum(call & (y == 1))); fp = int(np.sum(call & (y == 0)))
    fn = int(np.sum(~call & (y == 1))); tn = int(np.sum(~call & (y == 0)))
    profit = tp * (value - cost) - fp * cost
    everyone = int(y.sum()) * (value - cost) - int((y == 0).sum()) * cost
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=["predicted: don't call", "predicted: call"],
        y=["actual: no", "actual: yes"], text=[[tn, fp], [fn, tp]], texttemplate="%{text}",
        colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix @ threshold {threshold:.2f}", yaxis_autorange="reversed")

    grid = np.round(np.arange(0.01, 0.91, 0.01), 2)
    prof = [tp_ * (value - cost) - fp_ * cost for tp_, fp_ in
            ((int(np.sum((p >= t) & (y == 1))), int(np.sum((p >= t) & (y == 0)))) for t in grid)]
    curve = go.Figure(go.Scatter(x=grid, y=prof, mode="lines", name="model"))
    curve.add_hline(y=everyone, line_dash="dash", annotation_text="call everyone")
    curve.add_hline(y=0, line_color="gray", annotation_text="call nobody")
    curve.add_vline(x=threshold, line_color="red")
    curve.add_vline(x=cost / value, line_dash="dot", annotation_text="break-even cost/value")
    curve.update_layout(title="Campaign profit vs. threshold", xaxis_title="threshold",
                        yaxis_title="profit (EUR)")

    md = (f"### Profit: **€{profit:,.0f}**  \n"
          f"Calls made: **{tp + fp:,}** of {len(y):,} ({(tp + fp) / len(y):.1%}) · "
          f"subscriptions won: **{tp:,}** of {int(y.sum()):,} (recall {recall:.1%}) · "
          f"hit rate (precision): **{precision:.1%}**  \n"
          f"Call everyone: €{everyone:,.0f} · call nobody: €0 · "
          f"break-even probability = cost/value = {cost / value:.3f} · "
          f"threshold chosen on validation in train.py = {CFG['threshold']}")
    return md, cm, curve


# --- Tab 4: score one client -------------------------------------------------
def score_client(age, job, marital, education, default, housing, loan, contact, month,
                 campaign, pdays_never, pdays, previous, poutcome):
    row = pd.DataFrame([{
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan, "contact": contact,
        "month": month, "campaign": campaign, "pdays": 999 if pdays_never else pdays,
        "previous": previous, "poutcome": poutcome, "y": 0,
    }])
    p = {MODEL_NAMES[k]: float(v[0]) for k, v in predict_all(row).items()}
    ev = p["scikit-learn"] * VALUE_PER_SUBSCRIPTION - COST_PER_CALL
    decision = "CALL" if p["scikit-learn"] >= CFG["threshold"] else "DON'T CALL"
    return p, f"**{decision}** (threshold {CFG['threshold']}); expected value of the call ≈ €{ev:.2f}"


# --- Layout ------------------------------------------------------------------
with gr.Blocks(title="Term-deposit call targeting") as demo:
    gr.Markdown(
        "# Who should the call centre phone next?\n"
        "Logistic regression on UCI Bank Marketing, trained three ways "
        "(scikit-learn, manual PyTorch, nn.Module + optim) on a **chronological** "
        f"split. Train/val/test positive rates: {CFG['positive_rate']['train']:.1%} / "
        f"{CFG['positive_rate']['val']:.1%} / {CFG['positive_rate']['test']:.1%}. "
        f"C = {CFG['C']}, threshold = {CFG['threshold']}. Models loaded from `artifacts/`; nothing is retrained."
    )
    with gr.Tab("Model comparison"):
        split_dd = gr.Radio(["test", "val"], value="test", label="Split")
        table = gr.Dataframe(summary_table("test"), label="Metrics (from train.py)")
        agree = RESULTS["agreement"]
        gr.Markdown(
            f"**Agreement:** max |coef diff| vs sklearn: manual {agree['max_abs_coef_diff_manual_vs_sklearn']:.4f}, "
            f"nn {agree['max_abs_coef_diff_nn_vs_sklearn']:.4f} · max |prob diff| on test: "
            f"manual {agree['max_abs_proba_diff_manual_vs_sklearn_test']:.4f}, "
            f"nn {agree['max_abs_proba_diff_nn_vs_sklearn_test']:.4f} · same call/no-call decision: "
            f"{agree['decision_agreement_manual_vs_sklearn_test']:.2%} / "
            f"{agree['decision_agreement_nn_vs_sklearn_test']:.2%}"
        )
        model_dd = gr.Radio(list(MODEL_NAMES), value="sklearn", label="Model for calibration plot")
        with gr.Row():
            cal_plot = gr.Plot()
            roc_plot = gr.Plot()
        scat = gr.Plot(model_scatter("test"))
        with gr.Row():
            gr.Plot(loss_plot())
        gr.Plot(coef_plot())

        def _refresh(split, model_key):
            return (summary_table(split), *pred_vs_actual(split, model_key), model_scatter(split))

        for comp in (split_dd, model_dd):
            comp.change(_refresh, [split_dd, model_dd], [table, cal_plot, roc_plot, scat])
        demo.load(_refresh, [split_dd, model_dd], [table, cal_plot, roc_plot, scat])

    with gr.Tab("Distributions"):
        gr.Plot(target_over_time())
        with gr.Row():
            feat_dd = gr.Dropdown(DIST_FEATURES, value="age", label="Feature")
            by_split = gr.Checkbox(False, label="Colour by split instead of by target")
        with gr.Row():
            d1, d2 = gr.Plot(), gr.Plot()
        feat_dd.change(distribution, [feat_dd, by_split], [d1, d2])
        by_split.change(distribution, [feat_dd, by_split], [d1, d2])
        demo.load(distribution, [feat_dd, by_split], [d1, d2])

    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            thr = gr.Slider(0.01, 0.9, value=CFG["threshold"], step=0.01, label="Decision threshold")
            t_split = gr.Radio(["test", "val"], value="test", label="Split")
            t_model = gr.Radio(list(MODEL_NAMES), value="sklearn", label="Model")
        with gr.Row():
            cost_in = gr.Number(COST_PER_CALL, label="Cost per call (EUR)")
            value_in = gr.Number(VALUE_PER_SUBSCRIPTION, label="Value per subscription (EUR)")
        out_md = gr.Markdown()
        with gr.Row():
            cm_plot, profit_plot = gr.Plot(), gr.Plot()
        inputs = [thr, t_split, t_model, cost_in, value_in]
        for comp in inputs:
            comp.change(threshold_view, inputs, [out_md, cm_plot, profit_plot])
        demo.load(threshold_view, inputs, [out_md, cm_plot, profit_plot])

    with gr.Tab("Score a client"):
        lv = pre.levels_
        with gr.Row():
            c_age = gr.Slider(17, 98, value=40, step=1, label="age")
            c_job = gr.Dropdown(lv["job"], value="admin.", label="job")
            c_mar = gr.Dropdown(lv["marital"], value="married", label="marital")
            c_edu = gr.Dropdown(lv["education"], value="university.degree", label="education")
        with gr.Row():
            c_def = gr.Dropdown(["no", "unknown"], value="no", label="default")
            c_hou = gr.Dropdown(lv["housing"], value="yes", label="housing loan")
            c_loan = gr.Dropdown(lv["loan"], value="no", label="personal loan")
            c_con = gr.Dropdown(lv["contact"], value="cellular", label="contact")
            c_mon = gr.Dropdown(["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                                value="may", label="month of call")
        with gr.Row():
            c_camp = gr.Slider(1, 20, value=1, step=1, label="calls this campaign (incl. this one)")
            c_never = gr.Checkbox(True, label="never contacted before")
            c_pdays = gr.Slider(0, 30, value=5, step=1, label="days since last contact")
            c_prev = gr.Slider(0, 7, value=0, step=1, label="previous contacts")
            c_pout = gr.Dropdown(lv["poutcome"], value="nonexistent", label="previous outcome")
        btn = gr.Button("Score")
        c_out = gr.Label(label="P(subscribe) per model")
        c_md = gr.Markdown()
        btn.click(score_client, [c_age, c_job, c_mar, c_edu, c_def, c_hou, c_loan, c_con, c_mon,
                                 c_camp, c_never, c_pdays, c_prev, c_pout], [c_out, c_md])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)

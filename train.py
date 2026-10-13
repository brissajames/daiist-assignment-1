"""
Assignment 1 — training pipeline.

Predicts whether a bank client will subscribe to a term deposit if called
(UCI Bank Marketing), with logistic regression trained three ways:
scikit-learn, a manual PyTorch loop, and nn.Module + torch.optim.
Everything app.py needs is written to artifacts/.
"""

import json

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from pipeline import (
    ARTIFACTS, CATEGORICAL, COST_PER_CALL, MACRO, NUMERIC, VALUE_PER_SUBSCRIPTION,
    LogisticRegressionNet, Preprocessor, business_value, chronological_split,
    load_raw,
)

SEED = 42
C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0]
MANUAL_LR_GRID = [0.1, 0.5, 1.0, 2.0]
MANUAL_EPOCHS = 3000
ADAM_LR_GRID = [0.01, 0.05, 0.1]
ADAM_EPOCHS = 1500


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    # Precision among the top 10% scored clients: what a call centre with
    # capacity for 10% of the list would actually get.  Lift = vs. base rate.
    k = max(1, int(0.10 * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    prec10 = float(y[top].mean())
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "precision_top10": prec10,
        "lift_top10": prec10 / float(y.mean()),
    }


def fit_sklearn(X, y, C):
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


# ---------------------------------------------------------------------------
# The objective all three implementations minimise is the same:
#   sklearn:  0.5*||w||^2 + C * sum(logloss)
#   divide by C*n  ->  mean(logloss) + (1 / (2*C*n)) * ||w||^2
# so in PyTorch we use lam = 1/(C*n) and penalise only w (not the bias),
# exactly like sklearn does.
# ---------------------------------------------------------------------------

def objective(logits, y, w, lam):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    return bce + 0.5 * lam * torch.sum(w ** 2)


def fit_manual_torch(X, y, lam, lr, epochs=MANUAL_EPOCHS):
    """Session-5 style: raw tensors, autograd, hand-written gradient step."""
    torch.manual_seed(SEED)
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for _ in range(epochs):
        loss = objective(Xt @ w + b, yt, w, lam)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
        history.append(loss.item())
    return w.detach(), b.detach(), history


def fit_nn_torch(X, y, lam, lr, epochs=ADAM_EPOCHS):
    """Standard workflow: nn.Module + optimizer + loss."""
    torch.manual_seed(SEED)
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float32)
    model = LogisticRegressionNet(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = objective(model(Xt), yt, model.linear.weight, lam)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return model, history


def proba_manual(w, b, X):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(X) @ w + b).numpy()


def proba_nn(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def ablation(train, val, test, C):
    """Val (and, for the record only, test) metrics for feature-set choices.
    Decisions were made on validation; test is reported to be honest about
    whether those decisions held up."""
    variants = {
        "final feature set": (CATEGORICAL, NUMERIC),
        "+ macro indicators": (CATEGORICAL, NUMERIC + MACRO),
        "- month": ([c for c in CATEGORICAL if c != "month"], NUMERIC),
        "+ days_since_prev (0 = never)": (CATEGORICAL, NUMERIC + ["days_since_prev"]),
        "+ day_of_week": (CATEGORICAL + ["day_of_week"], NUMERIC),
        "raw age only (no age_band)": ([c for c in CATEGORICAL if c != "age_band"], NUMERIC),
        "raw pdays/campaign (no FE)": (
            CATEGORICAL,
            ["age", "pdays", "campaign", "previous"],
        ),
        "+ duration (LEAKAGE, not deployable)": (CATEGORICAL, NUMERIC + ["duration"]),
    }
    out = {}
    for name, (cats, nums) in variants.items():
        pre = Preprocessor(cats, nums).fit(train)
        m = fit_sklearn(pre.transform(train), train["y"].to_numpy(), C)
        out[name] = {
            split: metrics(d["y"].to_numpy(), m.predict_proba(pre.transform(d))[:, 1])
            for split, d in (("val", val), ("test", test))
        }
    return out


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df = load_raw()
    train, val, test = chronological_split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:5s}: {len(part):6d} rows, positive rate {part['y'].mean():.3f}")

    pre = Preprocessor().fit(train)
    X_tr, X_va, X_te = (pre.transform(d) for d in (train, val, test))
    y_tr, y_va, y_te = (d["y"].to_numpy() for d in (train, val, test))
    n = len(y_tr)
    print(f"{X_tr.shape[1]} features after encoding")

    # 1) Regularisation strength, chosen on validation PR-AUC (ranking of
    #    who to call matters more than calibration under drift).
    c_search = {}
    for C in C_GRID:
        p = fit_sklearn(X_tr, y_tr, C).predict_proba(X_va)[:, 1]
        c_search[str(C)] = metrics(y_va, p)
    best_C = float(max(c_search, key=lambda c: c_search[c]["pr_auc"]))
    lam = 1.0 / (best_C * n)
    print(f"best C = {best_C} (lambda = {lam:.2e})")

    # 2) The three implementations on the same split, same objective.
    sk = fit_sklearn(X_tr, y_tr, best_C)

    manual_search = {}
    for lr in MANUAL_LR_GRID:
        w, b, h = fit_manual_torch(X_tr, y_tr, lam, lr)
        manual_search[str(lr)] = {"final_train_objective": h[-1],
                                  "val": metrics(y_va, proba_manual(w, b, X_va))}
    best_manual_lr = float(min(manual_search,
                               key=lambda k: manual_search[k]["final_train_objective"]))
    w, b, manual_hist = fit_manual_torch(X_tr, y_tr, lam, best_manual_lr)

    nn_search = {}
    for lr in ADAM_LR_GRID:
        m, h = fit_nn_torch(X_tr, y_tr, lam, lr)
        nn_search[str(lr)] = {"final_train_objective": h[-1],
                              "val": metrics(y_va, proba_nn(m, X_va))}
    best_nn_lr = float(min(nn_search, key=lambda k: nn_search[k]["final_train_objective"]))
    nn_model, nn_hist = fit_nn_torch(X_tr, y_tr, lam, best_nn_lr)

    # sklearn's objective value, for an apples-to-apples convergence check
    with torch.no_grad():
        sk_obj = objective(torch.tensor(sk.decision_function(X_tr), dtype=torch.float32),
                           torch.tensor(y_tr, dtype=torch.float32),
                           torch.tensor(sk.coef_[0], dtype=torch.float32), lam).item()

    probas = {
        "val": {
            "sklearn": sk.predict_proba(X_va)[:, 1],
            "manual_torch": proba_manual(w, b, X_va),
            "nn_torch": proba_nn(nn_model, X_va),
        },
        "test": {
            "sklearn": sk.predict_proba(X_te)[:, 1],
            "manual_torch": proba_manual(w, b, X_te),
            "nn_torch": proba_nn(nn_model, X_te),
        },
    }
    # Naive baseline: every client gets the training base rate.
    base_rate = float(y_tr.mean())
    probas["val"]["baseline"] = np.full(len(y_va), base_rate)
    probas["test"]["baseline"] = np.full(len(y_te), base_rate)

    # 3) Decision threshold: maximise campaign profit on VALIDATION, then
    #    apply unchanged to test.  Break-even theory says cost/value.
    grid = np.round(np.arange(0.01, 0.91, 0.01), 2)
    val_profit = [business_value(y_va, probas["val"]["sklearn"], t)["profit"] for t in grid]
    best_t = float(grid[int(np.argmax(val_profit))])
    print(f"profit-maximising threshold on val = {best_t} "
          f"(break-even cost/value = {COST_PER_CALL / VALUE_PER_SUBSCRIPTION:.3f})")

    results = {}
    for split, y in [("val", y_va), ("test", y_te)]:
        results[split] = {}
        for name, p in probas[split].items():
            r = metrics(y, p)
            r["business_at_threshold"] = business_value(y, p, best_t)
            results[split][name] = r
        results[split]["call_everyone"] = {"business": business_value(y, np.ones(len(y)), 0.5)}

    coefs = {
        "sklearn": sk.coef_[0].tolist() + [float(sk.intercept_[0])],
        "manual_torch": w.tolist() + [float(b.item())],
        "nn_torch": nn_model.linear.weight.detach()[0].tolist()
                    + [float(nn_model.linear.bias.item())],
    }
    c = {k: np.array(v) for k, v in coefs.items()}
    agreement = {
        "max_abs_coef_diff_manual_vs_sklearn": float(np.max(np.abs(c["manual_torch"] - c["sklearn"]))),
        "max_abs_coef_diff_nn_vs_sklearn": float(np.max(np.abs(c["nn_torch"] - c["sklearn"]))),
        "max_abs_proba_diff_manual_vs_sklearn_test": float(np.max(np.abs(
            probas["test"]["manual_torch"] - probas["test"]["sklearn"]))),
        "max_abs_proba_diff_nn_vs_sklearn_test": float(np.max(np.abs(
            probas["test"]["nn_torch"] - probas["test"]["sklearn"]))),
        "decision_agreement_manual_vs_sklearn_test": float(np.mean(
            (probas["test"]["manual_torch"] >= best_t) == (probas["test"]["sklearn"] >= best_t))),
        "decision_agreement_nn_vs_sklearn_test": float(np.mean(
            (probas["test"]["nn_torch"] >= best_t) == (probas["test"]["sklearn"] >= best_t))),
        "final_objective": {"sklearn": sk_obj, "manual_torch": manual_hist[-1],
                            "nn_torch": nn_hist[-1]},
    }

    for split in ("val", "test"):
        print(f"\n{split} set:")
        print(f"{'model':14s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'logloss':>8s} {'P@10%':>7s} "
              f"{'calls':>6s} {'profit':>9s}")
        for name, r in results[split].items():
            if name == "call_everyone":
                bv = r["business"]
                print(f"{name:14s} {'':>8s} {'':>8s} {'':>8s} {'':>7s} {bv['calls']:6d} {bv['profit']:9.0f}")
            else:
                bv = r["business_at_threshold"]
                print(f"{name:14s} {r['roc_auc']:8.4f} {r['pr_auc']:8.4f} {r['log_loss']:8.4f} "
                      f"{r['precision_top10']:7.3f} {bv['calls']:6d} {bv['profit']:9.0f}")
    print("\nAgreement:", json.dumps({k: v for k, v in agreement.items()}, indent=2))

    abl = ablation(train, val, test, best_C)
    print("\nFeature ablation (PR-AUC / ROC-AUC):")
    for k, v in abl.items():
        print(f"  {k:38s} val {v['val']['pr_auc']:.4f} / {v['val']['roc_auc']:.4f}"
              f"   test {v['test']['pr_auc']:.4f} / {v['test']['roc_auc']:.4f}")

    # 4) Save everything the app needs. Nothing in app.py trains.
    joblib.dump(pre, ARTIFACTS / "preprocessor.joblib")
    joblib.dump(sk, ARTIFACTS / "sklearn_logreg.joblib")
    torch.save({"w": w, "b": b}, ARTIFACTS / "manual_torch.pt")
    torch.save(nn_model.state_dict(), ARTIFACTS / "nn_torch.pt")
    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump({
            "config": {
                "C": best_C, "lambda": lam, "manual_lr": best_manual_lr,
                "manual_epochs": MANUAL_EPOCHS, "adam_lr": best_nn_lr,
                "adam_epochs": ADAM_EPOCHS, "threshold": best_t,
                "base_rate_train": base_rate, "n_features": X_tr.shape[1],
                "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
                "positive_rate": {"train": float(y_tr.mean()), "val": float(y_va.mean()),
                                  "test": float(y_te.mean())},
            },
            "c_search": c_search, "manual_lr_search": manual_search,
            "adam_lr_search": nn_search, "ablation": abl,
            "results": results, "agreement": agreement,
            "feature_names": pre.feature_names_ + ["(intercept)"], "coefficients": coefs,
            "loss_history": {"manual_torch": manual_hist, "nn_torch": nn_hist},
        }, f, indent=2)
    print(f"\nSaved artifacts to {ARTIFACTS}")


if __name__ == "__main__":
    main()

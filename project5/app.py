"""
Project 5 - City Hospital: live training dashboard
Run:  pip install streamlit torch scikit-learn pandas numpy
      streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="City Hospital - Model Dashboard", layout="wide")

# ----------------------------------------------------------------------------
# Step 1: data (569 patients, 30 measurements, malignant = 1, split 60/20/20)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    d = load_breast_cancer()
    X = d.data.astype(np.float32)
    y = (d.target == 0).astype(np.float32)  # sklearn: 0 = malignant -> flip so malignant = 1
    return X, y


def make_splits(seed, scale_on):
    X, y = load_data()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    # Step 2: scaler fitted on TRAINING rows only (switchable)
    scaler = None
    if scale_on:
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_va, X_te = (scaler.transform(a).astype(np.float32) for a in (X_tr, X_va, X_te))

    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return {
        "scaler": scaler,
        "X_tr": t(X_tr), "y_tr": t(y_tr),
        "X_va": t(X_va), "y_va": t(y_va),
        "X_te": t(X_te), "y_te": t(y_te),
    }


# ----------------------------------------------------------------------------
# Step 3: training function (takes settings, reports both losses every epoch)
# ----------------------------------------------------------------------------
def build_model(n_in, hidden, layers, dropout):
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    mods.append(nn.Linear(d, 1))
    return nn.Sequential(*mods)


@torch.no_grad()
def eval_loss(model, X, y, loss_fn):
    model.eval()
    return loss_fn(model(X).squeeze(1), y).item()


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(X).squeeze(1)).numpy()


def train_model(cfg, data, on_epoch=None):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = build_model(data["X_tr"].shape[1], cfg["hidden"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(data["X_tr"])
    hist = {"train": [], "val": []}
    best_val, best_ep, bad = float("inf"), 0, 0

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss = loss_fn(model(data["X_tr"][idx]).squeeze(1), data["y_tr"][idx])
            loss.backward()
            opt.step()

        tr = eval_loss(model, data["X_tr"], data["y_tr"], loss_fn)
        va = eval_loss(model, data["X_va"], data["y_va"], loss_fn)
        hist["train"].append(tr)
        hist["val"].append(va)

        if va < best_val - 1e-6:
            best_val, best_ep, bad = va, ep, 0
        else:
            bad += 1
        if on_epoch:
            on_epoch(ep, hist)
        if cfg["patience"] > 0 and bad >= cfg["patience"]:
            break

    return model, hist, best_val


# ----------------------------------------------------------------------------
# Step 4: diagnosis rule
# ----------------------------------------------------------------------------
def diagnose(hist, best_val):
    final_train, final_val = hist["train"][-1], hist["val"][-1]
    if final_train > 0.3:
        return "underfitting"
    if final_val > 1.3 * best_val:
        return "overfitting"
    return "healthy"


def confusion(model, X, y, thr=0.5):
    p = predict_proba(model, X)
    pred = (p >= thr).astype(int)
    y = y.numpy().astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


# ----------------------------------------------------------------------------
# Step 5: page - sidebar settings, Train button, live chart, progress bar
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("runs", [])
ss.setdefault("model", None)
ss.setdefault("data", None)
ss.setdefault("cfg", None)
ss.setdefault("final", None)

st.title("City Hospital - Breast Cancer Model Dashboard")

with st.sidebar:
    st.header("Settings")
    scale_on = st.checkbox("Scale features", value=True)
    hidden = st.select_slider("Hidden units", [4, 8, 16, 32, 64, 128], value=16)
    layers = st.slider("Hidden layers", 1, 4, 2)
    lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], value=1e-3)
    epochs = st.slider("Max epochs", 5, 300, 100, step=5)
    batch = st.select_slider("Batch size", [8, 16, 32, 64, 128, 342], value=32)
    dropout = st.slider("Dropout", 0.0, 0.7, 0.0, step=0.05)
    wd = st.select_slider("Weight decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2], value=0.0)
    patience = st.slider("Early stop patience (0 = off)", 0, 50, 0)
    seed = st.number_input("Seed", 0, 9999, 42)
    train_btn = st.button("Train", type="primary", use_container_width=True)

chart_box = st.empty()
bar = st.progress(0.0)
result_box = st.container()

if train_btn:
    cfg = dict(scale=scale_on, hidden=hidden, layers=layers, lr=lr, epochs=epochs,
               batch=batch, dropout=dropout, wd=wd, patience=patience, seed=int(seed))
    data = make_splits(cfg["seed"], cfg["scale"])

    def on_epoch(ep, hist):
        bar.progress(ep / cfg["epochs"])
        if ep % 5 == 0 or ep == 1:  # redraw every 5 epochs
            chart_box.line_chart(pd.DataFrame({"train loss": hist["train"],
                                               "validation loss": hist["val"]}))

    with st.spinner("Training..."):
        model, hist, best_val = train_model(cfg, data, on_epoch)

    chart_box.line_chart(pd.DataFrame({"train loss": hist["train"],
                                       "validation loss": hist["val"]}))  # final redraw
    bar.progress(1.0)

    diagnosis = diagnose(hist, best_val)
    tp, fp, fn, tn = confusion(model, data["X_va"], data["y_va"])

    ss.model, ss.data, ss.cfg, ss.final = model, data, cfg, None  # new model -> new final test
    ss.last = dict(diagnosis=diagnosis, epochs_run=len(hist["train"]), best_val=best_val,
                   tp=tp, fp=fp, total_malignant=tp + fn)
    ss.runs.append({
        "run": len(ss.runs) + 1, **cfg,
        "epochs_run": len(hist["train"]),
        "final_train_loss": round(hist["train"][-1], 4),
        "final_val_loss": round(hist["val"][-1], 4),
        "best_val_loss": round(best_val, 4),
        "cancers_found": tp, "false_alarms": fp,
        "diagnosis": diagnosis,
    })

# ----------------------------------------------------------------------------
# Step 6: diagnosis + four numbers
# ----------------------------------------------------------------------------
if ss.get("last"):
    L = ss.last
    with result_box:
        msg = f"Diagnosis: **{L['diagnosis']}**"
        {"healthy": st.success, "overfitting": st.warning, "underfitting": st.error}[L["diagnosis"]](msg)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Epochs run", L["epochs_run"])
        c2.metric("Best validation loss", f"{L['best_val']:.4f}")
        c3.metric("Cancers found", f"{L['tp']} / {L['total_malignant']}")
        c4.metric("False alarms", L["fp"])
        st.caption("Cancers found and false alarms are counted on the validation set.")

# ----------------------------------------------------------------------------
# Step 7: run table + CSV, check a patient, one-time final test
# ----------------------------------------------------------------------------
st.divider()
st.subheader("All runs")
if ss.runs:
    runs_df = pd.DataFrame(ss.runs)
    st.dataframe(runs_df, use_container_width=True, hide_index=True)
    st.download_button("Download runs as CSV", runs_df.to_csv(index=False).encode(),
                       "runs.csv", "text/csv")
else:
    st.info("No runs yet. Set your options in the sidebar and press Train.")

left, right = st.columns(2)

with left:
    st.subheader("Check a patient")
    if ss.model is None:
        st.info("Train a model first.")
    else:
        n_val = len(ss.data["X_va"])
        i = st.number_input("Validation patient number", 0, n_val - 1, 0)
        prob = float(predict_proba(ss.model, ss.data["X_va"][i:i + 1])[0])
        actual = int(ss.data["y_va"][i].item())
        st.metric("Probability of malignant", f"{prob:.1%}")
        st.write(f"Model says: **{'malignant' if prob >= 0.5 else 'benign'}** | "
                 f"Actual: **{'malignant' if actual else 'benign'}**")

with right:
    st.subheader("Final test (one time only)")
    if ss.model is None:
        st.info("Train a model first.")
    elif ss.final is not None:
        F = ss.final
        st.success("Final test already used.")
        a, b, c = st.columns(3)
        a.metric("Test loss", f"{F['loss']:.4f}")
        b.metric("Accuracy", f"{F['acc']:.1%}")
        c.metric("Cancers found", f"{F['tp']} / {F['tp'] + F['fn']}")
        st.write(f"False alarms: **{F['fp']}** | Missed cancers: **{F['fn']}**")
    else:
        st.warning("Choose your final settings first. This can only be run once.")
        if st.button("Run final test"):
            d, m = ss.data, ss.model
            tp, fp, fn, tn = confusion(m, d["X_te"], d["y_te"])
            ss.final = dict(
                loss=eval_loss(m, d["X_te"], d["y_te"], nn.BCEWithLogitsLoss()),
                acc=(tp + tn) / (tp + fp + fn + tn), tp=tp, fp=fp, fn=fn, tn=tn)
            st.rerun()
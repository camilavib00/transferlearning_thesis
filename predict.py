import os, glob
import numpy as np
import pandas as pd
import torch
from train_rnn_chunks import RNNRegressor #importing model architecture


'''
Configuration Dictionaries
'''
DATAFILES = {
    "df15": "data/Exp3_Ins10_df15.xlsx",
    "df16": "data/Exp3_Ins100_df16.xlsx",
    "df18": "data/Exp4_Ins100_df18.xlsx",
    "df19": "data/Exp4_Ins10000_df19.xlsx",
}

MODEL_PATHS = [
    "models/[1]rnn_model_chunks_200e_15l_4_lr_64hidden.pth",
    "models/[2]rnn_model_chunks_200e_15l_4_lr_64hidden.pth",
    "models/[3]rnn_model_chunks_200e_15l_4_lr_64hidden.pth",
    "models/[4]rnn_model_chunks_200e_15l_4_lr_64hidden.pth"
]

FINETUNED_MODEL_PATHS = [
    "models/finetuned/[1]rnn_model_chunks_200e_15l_4_lr_64hidden__FT_df16.pth",
    "models/finetuned/[2]rnn_model_chunks_200e_15l_4_lr_64hidden__FT_df16.pth",
    "models/finetuned/[3]rnn_model_chunks_200e_15l_4_lr_64hidden__FT_df18.pth",
    "models/finetuned/[4]rnn_model_chunks_200e_15l_4_lr_64hidden__FT_df18.pth"
]

TOTAL_POINTS = 200
CTX_LEN      = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Helper functions for loading data, models, making predictions, and evaluating
'''
def resolve_path(pattern: str) -> str:
    if os.path.isfile(pattern):
        return pattern
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file found for: {pattern}")
    return hits[0]

def read_simbaml_seed(xlsx_path: str, time_col="time", value_col="observable", n=20) -> np.ndarray:
    df = pd.read_excel(xlsx_path)
    if time_col in df.columns:
        df = df.dropna(subset=[time_col, value_col]).sort_values(time_col)
    series = df[value_col].astype(np.float32).to_numpy()
    if len(series) < n:
        raise ValueError(f"{xlsx_path} has only {len(series)} rows, need at least {n}.")
    return series[:n]

def read_full_series(xlsx_path, time_col="time", value_col="observable"):
    df = pd.read_excel(xlsx_path)
    if time_col in df.columns:
        df = df.dropna(subset=[time_col, value_col]).sort_values(time_col)
        t = df[time_col].astype(np.float32).to_numpy()
    else:
        df = df.dropna(subset=[value_col])
        t = np.arange(len(df), dtype=np.float32)
    y = df[value_col].astype(np.float32).to_numpy()
    return t, y

def load_model(path: str) -> torch.nn.Module:
    m = RNNRegressor().to(device)  # must match training architecture
    state = torch.load(path, map_location=device)
    m.load_state_dict(state)
    m.eval()
    return m



'''
Autoregressive forecast using ONLY the seed (length ctx_len).
mode="block": append up to ctx_len outputs per step (fast, training-like)
mode="step" : append one output per step (slower, more realistic)
'''
@torch.no_grad()
def predict_from_seed(model: torch.nn.Module, seed: np.ndarray,
                      total_points=200, ctx_len=20, mode="step") -> np.ndarray:
    seed = np.asarray(seed, dtype=np.float32)
    assert len(seed) >= ctx_len, f"Seed must have >= {ctx_len} points."

    preds = list(map(float, seed[:ctx_len]))
    while len(preds) < total_points:
        context = preds[-ctx_len:]
        x = torch.tensor(context, dtype=torch.float32, device=device).view(1, ctx_len, 1)
        out = model(x)  # shape: (1, ctx_len)
        if mode == "block":
            block = out.squeeze(0).detach().cpu().numpy().astype(np.float32)
            take = min(ctx_len, total_points - len(preds))
            preds.extend(block[:take].tolist())
        else:
            # keep exactly as in your current script
            preds.append(float(out[:, 0].item()))
    return np.array(preds[:total_points], dtype=np.float32)

'''
Returns a dict with:
    - 'seed': (ctx_len,) seed array
    - 'preds_by_model': list[np.ndarray] length = len(model_paths)
    - 't_gt', 'y_gt': full ground-truth time & series from Simba-ML Excel
'''
def run_predictions_for(df_key: str,
                        model_paths=MODEL_PATHS,
                        total_points: int = TOTAL_POINTS,
                        ctx_len: int = CTX_LEN):
    xlsx = resolve_path(DATAFILES[df_key])
    seed = read_simbaml_seed(xlsx, n=ctx_len)
    t_gt, y_gt = read_full_series(xlsx)

    preds_by_model = []
    for p in model_paths:
        if not os.path.isfile(p):
            preds_by_model.append(None)
            continue
        model = load_model(p)
        preds_by_model.append(
            predict_from_seed(model, seed, total_points=total_points, ctx_len=ctx_len, mode="step")
        )

    return {
        "seed": seed,
        "preds_by_model": preds_by_model,
        "t_gt": t_gt,
        "y_gt": y_gt,
        "total_points": total_points,
        "ctx_len": ctx_len,
    }

'''
Helper functions for evaluation
Metrics: MAE, RMSE, R²
'''
def _safe_slice_forecast(y_true: np.ndarray, y_pred: np.ndarray, start_idx: int):
    """Return aligned slices from start_idx..end for fair post-seed evaluation."""
    n = min(len(y_true), len(y_pred))
    a = y_true[start_idx:n]
    b = y_pred[start_idx:n]
    return a, b

def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Classic definition; R² is undefined if y_true is constant
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot

'''
Compute MAE/RMSE/R² for the selected models vs. Simba‑ML GT,
evaluated ONLY on the forecast region (after the seed).
Returns a tidy DataFrame with one row per model.
'''
def evaluate_bundle(df_key: str,
                    bundle: dict,
                    models_to_use=None,
                    use_ctx_len: int | None = None) -> pd.DataFrame:
    if models_to_use is None:
        models_to_use = list(range(len(MODEL_PATHS)))
    ctx_len = int(bundle.get("ctx_len" if use_ctx_len is None else use_ctx_len, CTX_LEN))

    y_true = bundle["y_gt"]                      # Simba‑ML ground truth
    results = []
    for midx in models_to_use:
        pred = bundle["preds_by_model"][midx]
        if pred is None:
            continue
        yt, yp = _safe_slice_forecast(y_true, pred, start_idx=ctx_len)
        row = {
            "datafile": df_key,
            "model_id": midx + 1,               # human-friendly
            "n_eval_points": len(yt),
            "MAE": mae_score(yt, yp),
            "RMSE": rmse_score(yt, yp),
            "R2": r2_score(yt, yp),
        }
        results.append(row)
    return pd.DataFrame(results)

'''
Buiilds wide table with time, GT, and selected model predictions
'''
def build_series_table(df_key: str,
                       bundle: dict,
                       models_to_use=None) -> pd.DataFrame:
    if models_to_use is None:
        models_to_use = list(range(len(MODEL_PATHS)))
    t = bundle["t_gt"]
    y = bundle["y_gt"]
    frame = pd.DataFrame({"time": t, "GT_SimbaML": y})
    for midx in models_to_use:
        pred = bundle["preds_by_model"][midx]
        if pred is None:
            continue
        name = f"Pred_M{midx+1}"
        n = min(len(y), len(pred))
        frame[name] = np.nan
        frame.loc[:n-1, name] = pred[:n]
    return frame

'''
Write an Excel with:
    - 'Summary': MAE/RMSE/R² per model
    - 'Series' : time, GT, and predictions
custom_labels: list of str with same length as models_to_use
                (e.g. ["M1_base","M2_base","M1_FT","M2_FT"])
'''
def export_evaluation_excel(df_key: str,
                            bundle: dict,
                            out_path: str,
                            models_to_use=None,
                            custom_labels=None) -> str:
    if models_to_use is None:
        models_to_use = [i for i, p in enumerate(bundle["preds_by_model"]) if p is not None]

    if custom_labels is None:
        custom_labels = [f"M{m+1}" for m in models_to_use]

    summary_rows = []
    for m_idx, label in zip(models_to_use, custom_labels):
        pred = bundle["preds_by_model"][m_idx]
        if pred is None:
            continue
        yt, yp = _safe_slice_forecast(bundle["y_gt"], pred, start_idx=bundle["ctx_len"])
        row = {
            "datafile": df_key,
            "model_label": label,   # instead of just model_id
            "n_eval_points": len(yt),
            "MAE": mae_score(yt, yp),
            "RMSE": rmse_score(yt, yp),
            "R2": r2_score(yt, yp),
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    t = bundle["t_gt"]
    y = bundle["y_gt"]
    series = pd.DataFrame({"time": t, "GT_SimbaML": y})

    for m_idx, label in zip(models_to_use, custom_labels):
        pred = bundle["preds_by_model"][m_idx]
        if pred is None:
            continue
        n = min(len(y), len(pred))
        colname = f"Pred_{label}"
        series[colname] = np.nan
        series.loc[:n-1, colname] = pred[:n]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary")
        series.to_excel(writer, index=False, sheet_name="Series")
        # simple formatting nicety
        for sheet in ("Summary", "Series"):
            ws = writer.sheets[sheet]
            ws.set_column(0, series.shape[1], 16)
    return out_path


if __name__ == "__main__":
    DF_KEY = "df18"

    CUSTOM_PATHS = [
        "models/[3]rnn_model_chunks_200e_15l_4_lr_64hidden.pth",                    
        "models/[4]rnn_model_chunks_200e_15l_4_lr_64hidden.pth",                    
        "models/finetuned/[3]rnn_model_chunks_200e_15l_4_lr_64hidden__FT_df18.pth",
        "models/finetuned/[4]rnn_model_chunks_200e_15l_4_lr_64hidden__FT_df18.pth" 
    ]

    LABELS = ["M3_base", "M4_base", "M3_FT", "M4_FT"]

    bundle = run_predictions_for(DF_KEY, model_paths=CUSTOM_PATHS)

    out_xlsx = f"plots/eval_{DF_KEY}_customlabels.xlsx"
    export_evaluation_excel(
        DF_KEY,
        bundle,
        out_xlsx,
        models_to_use=[0,1,2,3],
        custom_labels=LABELS
    )
    print("Saved:", out_xlsx)
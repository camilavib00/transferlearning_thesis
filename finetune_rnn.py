import os
import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from train_rnn_chunks import RNNRegressor  # import model architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
Configuration Variables and general SetUp


'''

DF_KEY        = "df16"   #adapt to "df16" / "df18" / ...
POINTS_PATH   = "data/schwen_modeldata/model1_data16.xlsx" 
MODELS_TO_FT  = [1, 2]   # which models to fine-tune: 1..4
FT_EPOCHS     = 5        # 1 to maximum 5 epochs
LR            = 1e-4
SAVE_DIR      = "models/finetuned"
SHEET_NAME    = "Simulation"
TIME_COL      = "time"
VALUE_COL     = "Insulin_obs"

MODEL_PATHS = [
    "models/[1]rnn_model_chunks_200e_15l_4_lr_64hidden.pth",
    "models/[2]rnn_model_chunks_200e_15l_4_lr_64hidden.pth",
    "models/[3]rnn_model_chunks_200e_15l_4_lr_64hidden.pth",
    "models/[4]rnn_model_chunks_200e_15l_4_lr_64hidden.pth",
]

CTX_LEN   = 20
OUT_MAX   = 20
CLIP_NORM = 1.0
PATIENCE  = 1  # early stop if val doesn't improve

"""
Loads simulation points from an Excel file according to the specified format:
    times : (L,) float array of unique, sorted timestamps
    values: (num_series, L) float32 array (usually num_series=2)
Works for 2×13=26 or 2×16=32 rows (two series with same time stamps).
Also tolerates 1×13/16 (single series).
"""
def read_sim_points(path, sheet=SHEET_NAME, time_col=TIME_COL, value_col=VALUE_COL):

    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna(subset=[time_col, value_col]).copy()
    df[time_col] = df[time_col].astype(float)
    df[value_col] = df[value_col].astype(float)
    df = df.sort_values(time_col, kind="mergesort")  # stable to preserve pair order

    # unique times in order of appearance
    times_unique = pd.unique(df[time_col]).astype(float)
    L = len(times_unique)

    # collect values per timestamp (list per time)
    grouped = df.groupby(time_col)[value_col].apply(list)
    # how many series? -> maximum list length across timestamps
    num_series = int(grouped.map(len).max())

    if num_series not in (1, 2):
        raise ValueError(f"Expected 1 or 2 series per timestamp, got up to {num_series}.")

    # build (num_series, L) matrix in the order of times_unique
    values = np.full((num_series, L), np.nan, dtype=np.float32)
    for j, t in enumerate(times_unique):
        vals = grouped.loc[t]
        # if only one value at some t, duplicate for safety (keeps shapes consistent)
        if len(vals) == 1 and num_series == 2:
            vals = [vals[0], vals[0]]
        for s in range(num_series):
            values[s, j] = float(vals[s])

    return times_unique.astype(float), values.astype(np.float32)


def left_pad_to_ctx(seq, ctx_len):
    if len(seq) >= ctx_len:
        return seq[-ctx_len:]
    pad = np.full((ctx_len - len(seq),), seq[0], dtype=np.float32)
    return np.concatenate([pad, seq]).astype(np.float32)


"""
From a short series (length L=13 or 16) create small (x,y) pairs:
    for k=1..L-1:
    x_window = last ctx_len values up to k-1 (left-padded)
    y_future = series[k : min(L, k + out_max)]
"""
def one_step_blocks(series, ctx_len, out_max):
    L = len(series)
    pairs = []
    for k in range(1, L):
        x_win = left_pad_to_ctx(series[:k], ctx_len)
        y_fut = series[k: min(L, k + out_max)]
        if len(y_fut) == 0:
            continue
        pairs.append((x_win.astype(np.float32), y_fut.astype(np.float32)))
    return pairs


def masked_mse(pred, target):
    return torch.mean((pred - target) ** 2)


def run_epoch(model, optimizer, series_list, train: bool):
    model.train(mode=train)
    total_loss, total_pairs = 0.0, 0
    for series in series_list:
        pairs = one_step_blocks(series, CTX_LEN, OUT_MAX)
        for x_np, y_np in pairs:
            x = torch.tensor(x_np, dtype=torch.float32, device=device).view(1, CTX_LEN, 1)
            y = torch.tensor(y_np, dtype=torch.float32, device=device).view(1, -1)

            # temporarily adapt decoder length to current target window
            old_out = model.output_length
            model.output_length = y.shape[1]
            with torch.set_grad_enabled(train):
                pred = model(x)
                loss = masked_mse(pred, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                    optimizer.step()
            model.output_length = old_out

            total_loss += float(loss.item())
            total_pairs += 1
    return total_loss / max(1, total_pairs)


def rmse_from_mse(mse): 
    return math.sqrt(max(0.0, mse))


def freeze_rnn(model):
    for p in model.rnn.parameters():
        p.requires_grad = False


def finetune_one_model(base_path: str, df_key: str, times, values, epochs=FT_EPOCHS, lr=LR):
    # load base
    model = RNNRegressor().to(device)
    state = torch.load(base_path, map_location=device)
    model.load_state_dict(state, strict=True)

    freeze_rnn(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # split: series 0 -> train, series 1 -> val
    if values.shape[0] >= 2:
        train_series, val_series = [values[0]], [values[1]]
    else:
        train_series, val_series = [values[0]], [values[0][::-1]]

    best_val, best_state, no_improve = float("inf"), copy.deepcopy(model.state_dict()), 0

    for epoch in range(1, epochs + 1):
        train_mse = run_epoch(model, optimizer, train_series, train=True)
        val_mse   = run_epoch(model, optimizer, val_series,   train=False)
        print(f"[{os.path.basename(base_path)} | {df_key}] "
              f"Epoch {epoch}/{epochs} | train_rmse={rmse_from_mse(train_mse):.4f} "
              f"val_rmse={rmse_from_mse(val_mse):.4f}")
        if val_mse < best_val - 1e-9:
            best_val, best_state, no_improve = val_mse, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve > PATIENCE:
                print(" Early stopping.")
                break

    # save best model
    model.load_state_dict(best_state, strict=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    out_path = os.path.join(SAVE_DIR, f"{base_name}__FT_{df_key}.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved fine-tuned model: {out_path}")
    return out_path


if __name__ == "__main__":
    # load simulation points
    times, values = read_sim_points(POINTS_PATH, sheet=SHEET_NAME, time_col=TIME_COL, value_col=VALUE_COL)

    for mid in MODELS_TO_FT:
        idx = mid - 1  # 0-based
        if idx < 0 or idx >= len(MODEL_PATHS):
            print(f"Invalid model id: {mid}")
            continue
        base = MODEL_PATHS[idx]
        if not os.path.isfile(base):
            print(f"Missing model file: {base}")
            continue
        finetune_one_model(base, DF_KEY, times, values, epochs=FT_EPOCHS, lr=LR)
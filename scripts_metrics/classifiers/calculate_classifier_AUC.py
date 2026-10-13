import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pickle
from tqdm import tqdm

def load_emb(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        return obj.select_dtypes("number").values
    if isinstance(obj, dict):  # e.g. {"embeddings": ..., "labels": ...} or {id: vec}
        # adjust the key if needed
        obj = obj["embeddings"] if "embeddings" in obj else np.stack(list(obj.values()))
    if torch.is_tensor(obj):
        obj = obj.cpu().numpy()
    return np.asarray(obj)


def make_model(d_in, n_out):
    out = 1 if n_out == 2 else n_out
    if args.hidden == 0 or args.layers == 0:
        return nn.Linear(d_in, out)
    layers, d = [], d_in
    for _ in range(args.layers):
        layers += [nn.Linear(d, args.hidden), nn.ReLU(), nn.Dropout(args.dropout)]
        d = args.hidden
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)
 
 
def predict_proba(model, X_t):
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        if n_classes == 2:
            return torch.sigmoid(logits).squeeze(1).cpu().numpy()
        return torch.softmax(logits, dim=1).cpu().numpy()
 
 
def auc(y_true, probs):
    if n_classes == 2:
        return roc_auc_score(y_true, probs)
    return roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
 


p = argparse.ArgumentParser()
p.add_argument("--folds", type=int, default=5)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--hidden", type=int, default=256, help="0 = linear classifier")
p.add_argument("--layers", type=int, default=1)
p.add_argument("--dropout", type=int, default=0.1)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--random", action="store_true", help="skip training: randomly initialised model")
args = p.parse_args()
 
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
 
# ---- load data --
ref = load_emb("../embeddings/ref_embeddings_disease_ag_2048.pkl")
alt = load_emb("../embeddings/alt_embeddings_disease_ag_2048.pkl")
label_df = pd.read_csv("../data/variant_effects/variant_effects_disease.bed", sep="\t")
LABEL_COL = "label"  # adjust to the label column name in the .bed

assert ref.shape == alt.shape and len(ref) == len(label_df), (
    f"ref={ref.shape} alt={alt.shape} labels={len(label_df)}"
)

X = ((alt + ref)/2).astype(np.float32)
y_raw = label_df[LABEL_COL].values

classes, y = np.unique(y_raw, return_inverse=True)
n_classes = len(classes)
print(f"X: {X.shape} | classes: {n_classes} {dict(zip(classes, np.bincount(y)))} | device: {device}")
 
 
 
skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
aucs = []
 
for fold, (tr, te) in enumerate(skf.split(X, y), 1):
    # scale on train only (no leakage)
    scaler = StandardScaler().fit(X[tr])
    X_tr = torch.tensor(scaler.transform(X[tr]), dtype=torch.float32, device=device)
    X_te = torch.tensor(scaler.transform(X[te]), dtype=torch.float32, device=device)

    y_dtype = torch.float32 if n_classes == 2 else torch.long
    y_tr = torch.tensor(y[tr], dtype=y_dtype, device=device)
    y_te = torch.tensor(y[te], dtype=y_dtype, device=device)
    loss_fn = nn.BCEWithLogitsLoss() if n_classes == 2 else nn.CrossEntropyLoss()

    model = make_model(X.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if not args.random:
        n = len(X_tr)
        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(n, device=device)
            running, steps = torch.zeros((), device=device), 0
            for i in range(0, n, args.batch_size):
                idx = perm[i : i + args.batch_size]
                out = model(X_tr[idx])
                out = out.squeeze(1) if n_classes == 2 else out
                loss = loss_fn(out, y_tr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.detach()
                steps += 1
            print(f"Fold {fold} | epoch {epoch:3d} | train loss = {running.item() / steps:.4f}")

    # validation loss on the held-out fold
    model.eval()
    with torch.no_grad():
        val_out = model(X_te)
        val_out = val_out.squeeze(1) if n_classes == 2 else val_out
        val_loss = loss_fn(val_out, y_te).item()

    fold_auc = auc(y[te], predict_proba(model, X_te))
    aucs.append(fold_auc)
    print(f"Fold {fold}: AUC = {fold_auc:.4f} | val loss = {val_loss:.4f}\n")
 
print(f"\nMean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
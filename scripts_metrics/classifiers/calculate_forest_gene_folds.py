import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

def load_emb(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        return obj.select_dtypes("number").values
    if isinstance(obj, dict):  # e.g. {"embeddings": ..., "labels": ...} or {id: vec}
        # adjust the key if needed
        obj = obj["embeddings"] if "embeddings" in obj else np.stack(list(obj.values()))
    return np.asarray(obj)
 


p = argparse.ArgumentParser()
p.add_argument("--n_estimators", type=int, default=500)
p.add_argument("--max_depth", type=int, default=8)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
 
# ---- load data --
ref = load_emb("../../embeddings/ref_embeddings_disease_dnabert2_512.pkl")
alt = load_emb("../../embeddings/alt_embeddings_disease_dnabert2_512.pkl")
label_df = pd.read_csv("../../data/variant_effects/variant_effects_disease.bed", sep="\t")
LABEL_COL = "label"  # adjust to the label column name in the .bed

assert ref.shape == alt.shape and len(ref) == len(label_df), (
    f"ref={ref.shape} alt={alt.shape} labels={len(label_df)}"
)
print(f"ref={ref.shape} alt={alt.shape} labels={len(label_df)}")

X = ((alt + ref)/2).astype(np.float32)
y_raw = label_df[LABEL_COL].values

classes, y = np.unique(y_raw, return_inverse=True)
n_classes = len(classes)
print(f"X: {X.shape} | classes: {n_classes} {dict(zip(classes, np.bincount(y)))}")
 
chrom = label_df["chromosome"].astype(str).str.replace("chr", "").values
test_groups = [
    {"3","6","9","12","16","18","19","21"},
    {"2","5","11","14","17","20","22","X"},
    {"1","4","7","8","10","13","15"},
]

aucs = []
auprcs = []

for g, test_set in enumerate(test_groups, 1):

    te = np.where(np.isin(chrom, list(test_set)))[0]
    tr = np.where(~np.isin(chrom, list(test_set)))[0]

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
    )
    clf.fit(X[tr], y[tr])

    probs = clf.predict_proba(X[te])
    probs = probs.get() if hasattr(probs, "get") else np.asarray(probs)  # cupy -> numpy if needed
    if n_classes == 2:
        probs_auc = probs[:, 1]
    else:
        probs_auc = probs

    fold_auc = roc_auc_score(y[te], probs_auc)
    fold_auprc = average_precision_score(y[te], probs_auc)

    val_loss = log_loss(y[te], probs, labels=np.arange(n_classes))

    aucs.append(fold_auc)
    auprcs.append(fold_auprc)
    print(f"Group {g}: AUC = {fold_auc:.4f} | AUPRC = {fold_auprc:.4f} | baseline = {y[te].mean():.4f} | n_test = {len(te)}")

print(f"\nMean AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"Mean AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
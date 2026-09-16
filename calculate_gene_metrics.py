import pandas as pd
from scipy import spatial
from sklearn.metrics import roc_auc_score
import scanpy as sc
from scipy.stats import spearmanr, hypergeom

df =  pd.read_csv('./cosine_distance_gene_test.csv')
averaged_df = df.groupby("gene_name").agg(
    distance=("distance", "mean"),
    label=("label", "max")   
).reset_index()


adata = sc.datasets.pbmc3k()

sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
hvg_rank = adata.var['variances_norm'].rank(ascending=False)


seq_genes = set(averaged_df["gene_name"])
scrna_genes = set(adata.var_names)

common_genes = seq_genes & scrna_genes

print(f"Sequence data: {len(seq_genes)} genes")
print(f"scRNA-seq data: {len(scrna_genes)} genes")
print(f"Common: {len(common_genes)} genes")

seq_scores = averaged_df.set_index("gene_name")["distance"].loc[list(common_genes)]
var_scores = adata.var["variances_norm"].loc[list(common_genes)]

top_dist = set(seq_scores.nlargest(100).index)
top_var = set(var_scores.nlargest(100).index)

overlap = len(top_dist & top_var)

M = len(common_genes)  # background = shared gene universe
pval = hypergeom.sf(overlap - 1, M, len(top_var), len(top_dist))
print(f"Overlap: {overlap}/100, p-value: {pval:.4g}")
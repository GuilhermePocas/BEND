import pandas as pd
from scipy import spatial
from sklearn.metrics import roc_auc_score

df =  pd.read_csv('./cosine_similarities_hyena_128.csv')
averaged_df = df.groupby("gene_name").agg(
    distance=("distance", "mean"),
    label=("label", "max")   
).reset_index()

averaged_df = averaged_df[averaged_df["label"] == 1]

print(averaged_df.sort_values('distance',ascending=False).head(10))

print(averaged_df['distance'].max())
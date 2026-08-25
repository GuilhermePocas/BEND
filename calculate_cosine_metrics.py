import pandas as pd
from scipy import spatial
from sklearn.metrics import roc_auc_score

#separar por consequence
df =  pd.read_csv('./cosine_similarities_hyena_128.csv')
final_score = roc_auc_score(df['label'], df['distance'])

dfs_by_con = {
   con: sub_df
   for con, sub_df in df.groupby("Consequence")
}

scores_by_con = {}
for con, sub_df in dfs_by_con.items():
    sub_score = roc_auc_score(sub_df['label'], sub_df['distance'])
    scores_by_con[con] = sub_score

for con, scr in scores_by_con.items():
    print(f"Consequence: {con}, AUC Score: {scr}")
print(f"Final Score: {final_score}")
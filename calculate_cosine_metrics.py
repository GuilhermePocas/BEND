import pandas as pd
from scipy import spatial
from sklearn.metrics import roc_auc_score

df =  pd.read_csv('./cosine_similarities_ag_disease.csv')
score = roc_auc_score(df['label'], df['distance'])

print(score)
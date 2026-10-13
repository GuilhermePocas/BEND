import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import os
import glob

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

dirs = sorted(get_immediate_subdirectories("runs"))

for dir in dirs:

    if "nt" not in dir:
        continue

    subdirs = sorted(get_immediate_subdirectories(f"runs/{dir}"))
    
    for subdir in subdirs:

        df = pd.read_csv(f"runs/{dir}/{subdir}/cosine_distance_{subdir}.csv")

        result = spearmanr(df['logFC'].abs(), df['distance'])
        print(dir, f"{subdir} run | Spearman correlation: ", result.statistic, " | p-value: ", result.pvalue)

        result = kendalltau(df['logFC'].abs(), df['distance'])
        print(dir, f"{subdir} run | Kendall correlation: ", result.statistic, " | p-value: ", result.pvalue)
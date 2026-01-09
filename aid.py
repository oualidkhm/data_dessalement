# aid.py
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# ---------------- Quick Info ----------------
def quick_info(df, name='df', head=5, show_info=True, show_describe=True, show_missing=True):
    """
    Affiche des informations sans modifier le DataFrame.
    """
    print(f"=== Quick info: {name} ===")
    print("Shape:", df.shape)

    if show_info:
        buf = io.StringIO()
        df.info(buf=buf, memory_usage='deep')
        print(buf.getvalue())

    print("\nDtypes counts:")
    print(df.dtypes.value_counts())

    if show_describe:
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            print("\nDescribe (numeric):")
            display(df.select_dtypes(include=[np.number]).describe().T)
        if df.select_dtypes(include=['object','category']).shape[1] > 0:
            print("\nDescribe (object):")
            display(df.select_dtypes(include=['object','category']).describe().T)

    if show_missing:
        miss = df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        print("\nMissing values (cols with >0):")
        if miss.empty:
            print("None")
        else:
            display(miss)

    ndup = int(df.duplicated().sum())
    print("\nDuplicates:", ndup)

    try:
        mem = df.memory_usage(deep=True).sum()
        print(f"\nMemory usage: {mem/1024**2:.2f} MB")
    except Exception:
        pass

    if head:
        print(f"\nHead ({head} rows):")
        display(df.head(head))


# ---------------- Visuals ----------------
def histograms(df, cols=None, bins=30, max_plots=20):
    numeric = df.select_dtypes(include=[np.number])
    if cols is None:
        cols = numeric.columns.tolist()
    else:
        cols = [c for c in cols if c in numeric.columns]
    cols = cols[:max_plots]

    for col in cols:
        plt.figure()
        plt.hist(numeric[col].dropna(), bins=bins)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


def corr_matrix(df, method='pearson', annot=False, top_n=None, figsize=(10,10)):
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] == 0:
        print("No numeric columns to compute correlation.")
        return None

    if top_n is not None and top_n < numeric.shape[1]:
        var = numeric.var().sort_values(ascending=False)
        cols = var.head(top_n).index.tolist()
        numeric = numeric[cols]

    corr = numeric.corr(method=method)

    plt.figure(figsize=figsize)
    im = plt.imshow(corr.values, aspect='auto', interpolation='nearest')
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(f"Correlation matrix ({method})")
    if annot:
        for (i, j), val in np.ndenumerate(corr.values):
            plt.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6)
    plt.tight_layout()
    plt.show()
    return corr


def missing_summary_plot(df):
    miss_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0]
    if miss_pct.empty:
        print("No missing values.")
        return miss_pct

    plt.figure(figsize=(8, max(4, 0.25 * len(miss_pct))))
    plt.barh(miss_pct.index.astype(str), miss_pct.values)
    plt.xlabel("Missing %")
    plt.title("Columns with missing values (%)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    return miss_pct


def quick_visuals(df, hist=True, corr=True, missing=True, hist_max=12, corr_top_n=30):
    """
    Tout en une fois: histogrammes, matrice de corrélation et missing summary.
    """
    corr_df = None
    miss = None

    if hist:
        print("-> Generating histograms (first columns up to hist_max)...")
        histograms(df, max_plots=hist_max)

    if corr:
        print("-> Generating correlation matrix (top columns by variance)...")
        corr_df = corr_matrix(df, top_n=corr_top_n)

    if missing:
        print("-> Generating missing values summary...")
        miss = missing_summary_plot(df)

    return corr_df, miss

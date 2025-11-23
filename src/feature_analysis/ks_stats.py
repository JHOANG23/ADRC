import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from src.config import FEATURE_PATH

def ks_nc_vs_mci(
    df: pd.DataFrame,
    label_col: str = "label",
    control_label: int = 2,
    min_valid: int = 2,
    return_p: bool = False,
) -> pd.DataFrame:
    """
    Compute KS statistic comparing NC (label=2) vs MCI+AD (label=1 or 0 combined).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe after merging your feature sets.
    label_col : str, default "label"
        Column containing diagnostic classes.
    control_label : int, default 2
        NC class code.
    min_valid : int, default 2
        Minimum samples per group to compute KS.
    return_p : bool, default False
        Whether to include p-values.
    drop_cols : list[str] or None
        Optional columns to drop before KS (e.g., file_name, language).

    Returns
    -------
    pd.DataFrame
        Feature, KS statistic (and p-value), sorted descending.
    """

    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not found in dataset.")

    work = df.copy()

    # Select numeric feature columns only
    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in num_cols:
        num_cols.remove(label_col)

    # Masks for the two groups
    nc_mask = work[label_col] == control_label        # NC = 2
    mci_ad_mask = work[label_col].isin([0, 1])        # MCI + AD combined

    rows = []
    for col in num_cols:
        x = work.loc[nc_mask, col].dropna()
        y = work.loc[mci_ad_mask, col].dropna()

        # Require minimum samples
        if len(x) < min_valid or len(y) < min_valid:
            stat, pval = np.nan, np.nan
        else:
            res = ks_2samp(x, y)
            stat, pval = float(res.statistic), float(res.pvalue)

        row = {"feature": col, "KS_nc-vs-mci_ad": stat}
        if return_p:
            row["pvalue"] = pval
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("KS_nc-vs-mci_ad", ascending=False)
    return out.reset_index(drop=True)

def main():
    # label: 2=NC, 1=MCI, 0=AD
    # Format dataframe for feature analysis
    df_adrc_egemaps = pd.read_csv(f"{FEATURE_PATH}/egemapsv02_equal_n1/features.csv")
    df_adrc_prosodic = pd.read_csv(f"{FEATURE_PATH}/praat_prosodic/features.csv")
    df_adrc_pyaudio = pd.read_csv(f"{FEATURE_PATH}/pyAudio/features.csv")
    df_adrc_egemaps = df_adrc_egemaps[df_adrc_egemaps['language'] == 'en'].drop(columns=['file_name', 'language'])
    df_adrc_prosodic = df_adrc_prosodic[df_adrc_prosodic['language'] == 'en'].drop(columns=['file_name', 'language'])
    df_adrc_pyaudio = df_adrc_pyaudio[df_adrc_pyaudio['language'] == 'en'].drop(columns=['file_name', 'language'])

    # df_adrc = df_adrc_egemaps.merge(df_adrc_prosodic, how='inner', on=['file_name', 'language', 'label'])
    # df_adrc = df_adrc.merge(df_adrc_pyaudio, how='inner', on=['file_name', 'language', 'label'])

    results = ks_nc_vs_mci(df_adrc_pyaudio, return_p=True)
    print(results.head(20))

if __name__ == "__main__":
    main()
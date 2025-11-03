import os
import torch
import pandas as pd
from utils import format_df
from egemaps_extract import egemaps_extract
from praat_extract import extract_prosodic_features
from config import TRIMMED_AUDIO_PATH, EXCEL_FULL_ADRC_PATH, FEATURE_PATH

def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    torch.set_printoptions(profile="full")

    # Calculate prosodic feature values and output labels
    df = format_df(EXCEL_FULL_ADRC_PATH)
    # praat_prosodic_X, praat_prosodic_Y = extract_prosodic_features(df, FEATURE_PATH)
    # torch.save(praat_prosodic_X, f"{FEATURE_PATH}/praat_prosodic/X.pt") # Dim (241, 18) (last 25 rows are spanish)
    # torch.save(praat_prosodic_Y, f"{FEATURE_PATH}/praat_prosodic/Y.pt") # Dim (241,) (last 25 rows are spanish)

    egemaps_X, egemaps_Y = egemaps_extract(df, TRIMMED_AUDIO_PATH, False)
    torch.save(egemaps_X, f"{FEATURE_PATH}/egemapsv02/X.pt") # Dim (241, 88) (last 25 rows are spanish)
    torch.save(egemaps_Y, f"{FEATURE_PATH}/egemapsv02/Y.pt") # Dim (241) (last 25 rows are spanish)

if __name__ == "__main__":
    main()
import os
import sys
import csv
import torch
import pandas as pd
from src.utils import format_df
from src.data_split import balance_data
from sklearn.model_selection import train_test_split
from src.feature_extraction.model_embed import embed_audio
from src.feature_extraction.pyAudio_extract import pyaudio_extract
from src.feature_extraction.egemaps_extract import egemaps_extract
from src.feature_extraction.praat_extract import extract_prosodic_features
from src.config import TRIMMED_AUDIO_PATH, EXCEL_FULL_ADRC_PATH, FEATURE_PATH, MODEL_PATH

def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    torch.set_printoptions(profile="full")
 
    df = format_df(EXCEL_FULL_ADRC_PATH)

    # egemaps extraction
    # mode = 'equal'
    # N = 1
    # csv_out = f"{FEATURE_PATH}/egemapsv02_{mode}_n{N}/features.csv"
    # csv_rows = egemaps_extract(
    #     df=df, 
    #     input_dir=TRIMMED_AUDIO_PATH, 
    #     mode=mode, 
    #     N=N,
    #     append_cov=False,
    # )
    # df_features = pd.DataFrame(csv_rows)
    # df_features.to_csv(csv_out, index=False)

    #praat extraction
    # csv_out = f"{FEATURE_PATH}/praat_prosodic/features.csv"
    # praat_csv_rows = extract_prosodic_features(df)
    # praat_df_features = pd.DataFrame(praat_csv_rows)
    # praat_df_features.to_csv(csv_out, index=False)

    # pyAudio extraction
    csv_out = f"{FEATURE_PATH}/pyAudio/features.csv"
    os.makedirs(f"{FEATURE_PATH}/pyAudio", exist_ok=True)
    pyaudio_csv_rows = pyaudio_extract(df, TRIMMED_AUDIO_PATH, 'equal', 1)
    df_features = pd.DataFrame(pyaudio_csv_rows)
    df_features.to_csv(csv_out, index=False)

    # os.makedirs(f"{FEATURE_PATH}/xlsr53", exist_ok=True)
    # X_en_tensor, Y_en_tensor, X_es_tensor, Y_es_tensor = embed_audio(df, TRIMMED_AUDIO_PATH, MODEL_PATH, batch_size=1)
    # torch.save(X_en_tensor, f"{FEATURE_PATH}/xlsr53/X_en.pt") # Dim (241*N, 88) (last 25 * SEG_NUM rows are spanish)
    # torch.save(Y_en_tensor, f"{FEATURE_PATH}/xlsr53/Y_en.pt") # Dim (241*N) (last 25 * SEG_NUM rows are spanish)
    # torch.save(X_es_tensor, f"{FEATURE_PATH}/xlsr53/X_es.pt")
    # torch.save(Y_es_tensor, f"{FEATURE_PATH}/xlsr53/Y_es.pt")

if __name__ == "__main__":
    main()
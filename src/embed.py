import os
import sys
import csv
import torch
import pandas as pd
from src.utils import format_df
from src.data_split import balance_data
from sklearn.model_selection import train_test_split
from src.feature_extraction.model_embed import embed_audio
from src.feature_extraction.egemaps_extract import egemaps_extract
from src.feature_extraction.praat_extract import extract_prosodic_features
from src.config import TRIMMED_AUDIO_PATH, EXCEL_FULL_ADRC_PATH, FEATURE_PATH, MODEL_PATH

def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    torch.set_printoptions(profile="full")

    # move this into validation script pipeline. 
    # moving forward, create embeddings first, with corresponding csv containing the file names of each input. 
    # this way, during validation, I simply sort the file names and their corresponding feature vector into the
    # the correct split
    df = format_df(EXCEL_FULL_ADRC_PATH)
    en_df = df[df['language'] == 'en']
    balanced_en_df = balance_data(en_df)
    y_df = balanced_en_df['synd2']
    train_en_X, val_en_X = train_test_split(
        balanced_en_df, 
        test_size=0.2, 
        train_size=0.8,
        stratify=y_df,
        random_state=42
    )

    # Calculate prosodic feature values and output labels
    # X_en_tensor, Y_en_tensor, X_es_tensor, Y_es_tensor = extract_prosodic_features(df, FEATURE_PATH)
    # torch.save(X_en_tensor, f"{FEATURE_PATH}/praat_prosodic/X_en.pt") # Dim (241, 18) (last 25 rows are spanish)
    # torch.save(Y_en_tensor, f"{FEATURE_PATH}/praat_prosodic/Y_en.pt") # Dim (241,) (last 25 rows are spanish)
    # torch.save(X_es_tensor, f"{FEATURE_PATH}/praat_prosodic/X_es.pt")
    # torch.save(Y_es_tensor, f"{FEATURE_PATH}/praat_prosodic/Y_es.pt")
    
    
    mode = 'equal'
    N = 10
    csv_out = f"{FEATURE_PATH}/egemapsv02_{mode}_n{N}/corresponding_files.csv"
    os.makedirs(f"{FEATURE_PATH}/egemapsv02_{mode}_n{N}", exist_ok=True)
    X_en_tensor, Y_en_tensor, X_es_tensor, Y_es_tensor, corresponding_files = egemaps_extract(df, TRIMMED_AUDIO_PATH, mode, N)
    torch.save(X_en_tensor, f"{FEATURE_PATH}/egemapsv02_{mode}_n{N}/X_en.pt") # Dim (241*N, 88) (last 25 * SEG_NUM rows are spanish)
    torch.save(Y_en_tensor, f"{FEATURE_PATH}/egemapsv02_{mode}_n{N}/Y_en.pt") # Dim (241*N) (last 25 * SEG_NUM rows are spanish)
    torch.save(X_es_tensor, f"{FEATURE_PATH}/egemapsv02_{mode}_n{N}/X_es.pt")
    torch.save(Y_es_tensor, f"{FEATURE_PATH}/egemapsv02_{mode}_n{N}/Y_es.pt")

    pd.DataFrame({"file_name": corresponding_files['en']}) \
    .to_csv(csv_out.replace(".csv","_en.csv"), index=False)

    pd.DataFrame({"file_name": corresponding_files['es']}) \
    .to_csv(csv_out.replace(".csv","_es.csv"), index=False)

    # os.makedirs(f"{FEATURE_PATH}/xlsr53", exist_ok=True)
    # X_en_tensor, Y_en_tensor, X_es_tensor, Y_es_tensor = embed_audio(df, TRIMMED_AUDIO_PATH, MODEL_PATH, batch_size=1)
    # torch.save(X_en_tensor, f"{FEATURE_PATH}/xlsr53/X_en.pt") # Dim (241*N, 88) (last 25 * SEG_NUM rows are spanish)
    # torch.save(Y_en_tensor, f"{FEATURE_PATH}/xlsr53/Y_en.pt") # Dim (241*N) (last 25 * SEG_NUM rows are spanish)
    # torch.save(X_es_tensor, f"{FEATURE_PATH}/xlsr53/X_es.pt")
    # torch.save(Y_es_tensor, f"{FEATURE_PATH}/xlsr53/Y_es.pt")

if __name__ == "__main__":
    main()
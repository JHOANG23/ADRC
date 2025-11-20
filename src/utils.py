import os
import torch
import audiofile
import pandas as pd
from src.config import TRIMMED_AUDIO_PATH, EXCEL_FULL_ADRC_PATH, FEATURE_PATH

# Performs a left join between two dataframes to append data for the feature extraction/training step. 
def format_df(excel_path: str) -> pd.DataFrame:
    def _unify_date(series):
        dt = pd.to_datetime(series, format='%m.%d.%Y', errors='coerce')
        dt = dt.fillna(pd.to_datetime(series, format='%m.%d.%y', errors='coerce'))
        return dt
    
    df = pd.read_excel(
        excel_path, 
        usecols=['adcid', 'date_str','age_at_recording','Gender','Education','synd2','synd5'], 
        dtype={'adcid':'string','Gender': 'Int64', 'Education': 'Int64', 'age_at_recording': 'Int64'}
    )
    df['date_str'] = _unify_date(df['date_str'])
    df['synd2'] = df['synd2'].astype('category')
    df['synd2'] = df['synd2'].cat.codes.astype('int64')

    rows = []
    for lang in ['en','es']:
        files = os.listdir(os.path.join(TRIMMED_AUDIO_PATH, lang))
        rows.append(pd.DataFrame({'file_name': files, 'adcid': files, 'date_str': files, 'language': lang}))
    adrc_df = pd.concat(rows, ignore_index=True)

    pattern = r'^(\d+)(?:trt)?\.(\d+\.\d+\.\d+)(?:_trimmed)?\..*$'
    adrc_df[['adcid','date_str']] = adrc_df['file_name'].str.extract(pattern)
    adrc_df['date_str'] = _unify_date(adrc_df['date_str'])
    adrc_df = pd.merge(adrc_df, df, on=['date_str', 'adcid'])

    return adrc_df

def split_wav(wav_file: str, mode: str, N: int):
    signal, sr = audiofile.read(wav_file, always_2d=True)
    if signal.ndim == 2:
        signal = signal.mean(axis=0)

    total_samples = signal.shape[0]
    segments = []
    if mode == 'equal':
        total_samples = (total_samples // N) * N
        signal = signal[:total_samples]
        seg_len = (total_samples // N)
        for i in range(N):
            start = i*seg_len
            end = (i+1) * seg_len if i < N - 1 else total_samples
            segments.append(signal[start:end])
    elif mode == 'seconds':
        seg_len = int(N * sr)
        start = 0
        while start < total_samples:
            end = min(start + seg_len, total_samples)
            segments.append(signal[start:end])
            start = end
    else:
        raise ValueError(f"Invalid mode input: {mode}. Must be 'equal' or 'seconds'")
    return segments, sr

# temporary function i made because I forgot to separate languages of feature tensor
def split_lang_pt(path, n_spanish=25):
    """
    path = directory containing X.pt and Y.pt
    n_spanish = last n samples that are spanish
    """

    X = torch.load(os.path.join(path, "X.pt"))
    Y = torch.load(os.path.join(path, "Y.pt"))

    X_en = X[:-n_spanish]
    Y_en = Y[:-n_spanish]
    X_es = X[-n_spanish:]
    Y_es = Y[-n_spanish:]

    torch.save(X_en, os.path.join(path, "X_en.pt"))
    torch.save(Y_en, os.path.join(path, "Y_en.pt"))
    torch.save(X_es, os.path.join(path, "X_es.pt"))
    torch.save(Y_es, os.path.join(path, "Y_es.pt"))

    print(f"saved in: {path}")
    print(f"English size: {len(X_en)} | Spanish size: {len(X_es)}")

def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    df = format_df(EXCEL_FULL_ADRC_PATH)
    print(df)
    en_df = df[df['language']=='en']
    es_df = df[df['language']=='es']
    en_class_counts = en_df['synd2'].value_counts().sort_index()
    es_class_counts = es_df['synd2'].value_counts().sort_index()

    print("Number of samples per english class:")
    for cls, count in en_class_counts.items():
        print(f"Class {cls}: {count}")
    
    print("Number of samples per english class:")
    for cls, count in es_class_counts.items():
        print(f"Class {cls}: {count}")

    embeddings = torch.load(f"{FEATURE_PATH}/egemapsv02_equal_n10/X_en.pt")
    print(embeddings.shape)

    # path = os.path.join(FEATURE_PATH, 'praat_prosodic')
    # split_lang_pt(path)

if __name__ == '__main__':
    main()

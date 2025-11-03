import os
import pandas as pd
from config import TRIMMED_AUDIO_PATH, EXCEL_FULL_ADRC_PATH

# Performs a left join between two dataframes to append data for the feature extraction/training step. 
def format_df(excel_path):
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
    adrc_df = pd.merge(adrc_df, df, on=['date_str', 'adcid'], how='left')

    return adrc_df

def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    df = format_df(EXCEL_FULL_ADRC_PATH)
    print(df)

if __name__ == '__main__':
    main()

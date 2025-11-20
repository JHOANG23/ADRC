import torch
import pandas as pd
from src.utils import format_df
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from src.config import TRIMMED_AUDIO_PATH, EXCEL_FULL_ADRC_PATH, FEATURE_PATH, MODEL_PATH


class MixedBatchIterator:
    """
    A custom iterable that wraps the English DataLoader and dynamically
    injects Spanish samples at a 4:1 ratio (every fifth sample is Spanish).
    """
    def __init__(self, en_loader: DataLoader, es_samples: TensorDataset, batch_size: int = 32):
        self.en_loader = en_loader
        self.en_iterator = iter(en_loader)
        
        # Spanish samples are a small, static set that will be repeatedly injected.
        self.es_X = es_samples.tensors[0]
        self.es_Y = es_samples.tensors[1]
        self.es_size = len(self.es_X)
        self.batch_size = batch_size
        
        # Index to cycle through the small Spanish sample set
        self.es_index = 0
        
    def __iter__(self):
        self.en_iterator = iter(self.en_loader)
        self.es_index = 0
        return self

    def __next__(self):
        try:
            en_x, en_y = next(self.en_iterator)
        except StopIteration:
            raise StopIteration


        num_es_injections = self.batch_size // 5 
        es_x_injections = []
        es_y_injections = []
        
        # Cycle through the small Spanish set for injections
        for _ in range(num_es_injections):
            es_x_injections.append(self.es_X[self.es_index])
            es_y_injections.append(self.es_Y[self.es_index])
            self.es_index = (self.es_index + 1) % self.es_size
            
        es_x_injections = torch.stack(es_x_injections)
        es_y_injections = torch.stack(es_y_injections)
        
        mixed_x = []
        mixed_y = []
        
        # Iterate over the target batch size
        en_idx = 0
        es_inj_idx = 0
        for i in range(self.batch_size):
            if (i + 1) % 5 == 0 and es_inj_idx < num_es_injections:
                mixed_x.append(es_x_injections[es_inj_idx].unsqueeze(0))
                mixed_y.append(es_y_injections[es_inj_idx].unsqueeze(0))
                es_inj_idx += 1
            else:
                # Add English sample
                if en_idx < en_x.shape[0]: # safety check for batch size mismatch
                    mixed_x.append(en_x[en_idx].unsqueeze(0))
                    mixed_y.append(en_y[en_idx].unsqueeze(0))
                    en_idx += 1
        
        final_x = torch.cat(mixed_x, dim=0)
        final_y = torch.cat(mixed_y, dim=0)

        return final_x, final_y
    
    def __len__(self):
        return len(self.en_loader)

def balance_data(df: pd.DataFrame, seed):
    # filter dataset to remove rows with no classification and fill in empty education with default value of 12
    filtered_df = df.dropna(subset=['synd2', 'age_at_recording', 'Gender']).copy()
    filtered_df = filtered_df[filtered_df['synd2'] != -1]
    filtered_df.loc[filtered_df['Education'].isna(), 'Education'] = 12
                    
    class_counts = filtered_df['synd2'].value_counts()
    print(class_counts)
    min_class_count = class_counts.min()
    
    normal_df = filtered_df[filtered_df['synd2']==2]
    mci_df = filtered_df[filtered_df['synd2']==1]
    dem_df = filtered_df[filtered_df['synd2']==0]

    bal_normal_df = normal_df.sample(min_class_count, random_state=seed)
    bal_mci_df = mci_df.sample(min_class_count, random_state=seed)
    bal_dem_df = dem_df.sample(min_class_count, random_state=seed)

    full_bal_df = pd.concat([bal_normal_df, bal_mci_df, bal_dem_df], axis = 0)
    full_bal_df = full_bal_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # normal_df = filtered_df[filtered_df['synd2']==0]
    # mci_df = filtered_df[filtered_df['synd2']==1]

    # bal_normal_df = normal_df.sample(min_class_count, random_state=seed)
    # bal_mci_df = mci_df.sample(min_class_count, random_state=seed)
    # full_bal_df = pd.concat([bal_normal_df, bal_mci_df], axis = 0)
    # full_bal_df = full_bal_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return full_bal_df

def custom_balanced_2_fold(df: pd.DataFrame, seed):
    #df is spanish
    nc_df = df[df['binary_label']==1].reset_index(drop=True)      # ADRC
    mci_ad_df = df[df['binary_label']==0].reset_index(drop=True)  # ADRC-M
    # nc_df = df[df['synd2']==0].reset_index(drop=True)      # ADRRES-M
    # mci_ad_df = df[df['synd2']==1].reset_index(drop=True)  # ADRRES-M

    nc_fold0 = nc_df.iloc[0::2]
    nc_fold1 = nc_df.iloc[1::2]
    mci_ad_fold0 = mci_ad_df.iloc[0::2]
    mci_ad_fold1 = mci_ad_df.iloc[1::2]

    # fold 0
    fold0_train = pd.concat([nc_fold1, mci_ad_fold1]).sample(frac=1, random_state=seed).reset_index(drop=True)
    fold0_val   = pd.concat([nc_fold0, mci_ad_fold0]).sample(frac=1, random_state=seed).reset_index(drop=True)
    # fold1
    fold1_train = pd.concat([nc_fold0, mci_ad_fold0]).sample(frac=1, random_state=seed).reset_index(drop=True)
    fold1_val   = pd.concat([nc_fold1, mci_ad_fold1]).sample(frac=1, random_state=seed).reset_index(drop=True)

    return fold0_train, fold1_train, fold0_val, fold1_val

def main():
    df = format_df(EXCEL_FULL_ADRC_PATH)
    es_df = df[df['language']=='es']
    balance_data(es_df)
    pass

if __name__ == "__main__":
    main()
import os
import torch
import shutil
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from src.utils import format_df
from src.data_split import MixedBatchIterator
from src.config import FEATURE_PATH, EXCEL_FULL_ADRC_PATH, CHECKPOINT_PATH

FEATURE_PATH = '/home/jobe/datasets/ADReSS-M/features'
CHECKPOINT_PATH = '/home/jobe/datasets/ADReSS-M/weights'

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=12, dropout=0.2):
        super().__init__()
        
        self.bn = nn.BatchNorm1d(input_dim)
        self.down_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_dim, 1)
        )
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.bn(x)
        h = x.permute(0,2,1)
        h = self.down_projection(h)
        a = self.attn_pool(h)
        w = torch.softmax(a, dim=1)
        pooled = (w * h).sum(dim=1)
        return self.output(pooled)
    
def train(device, model, loss_fn, optimizer, num_epoch, train_loader, val_loader, checkpoint_path, seed):
    val_losses = []
    min_val_loss = float('inf')
    best_checkpoint_file = ''
    global_step = 0
    default_lr = 3e-3
    warmup_steps = 100

    # train
    for epoch in range(num_epoch):
        model.train()
        total_train_loss = 0
        for x, y_true in train_loader:
            global_step+=1
            if global_step < warmup_steps:
                lr = default_lr * (global_step / warmup_steps)
            else:
                lr = default_lr  # stays 3e-3
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            x = x.to(device)
            y_true = y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y_true) 
            loss.backward()
            optimizer.step()
            total_train_loss+=loss.item()
        avg_train_loss = total_train_loss/len(train_loader)

        # validation after epoch
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y_true in val_loader:
                x = x.to(device)
                y_true = y_true.to(device)
                y_pred = model(x)
                val_loss = loss_fn(y_pred, y_true)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Save min loss
        if avg_val_loss < min_val_loss:
            if best_checkpoint_file:
                old_path = os.path.join(checkpoint_path, best_checkpoint_file)
                os.remove(old_path)

            min_val_loss = avg_val_loss
            filename = f"run_seed{seed}_epoch_{epoch+1}_weight.pt"
            full_save_path = os.path.join(checkpoint_path, filename)
            torch.save(model.state_dict(), full_save_path)
            best_checkpoint_file = filename
        # print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return pd.DataFrame({'checkpoint_file': [best_checkpoint_file], 'val_loss': [min_val_loss]})

def pre_train(device, embeddings, labels, en_train_df, es_val_loader, output_path):
    embeddings_en = embeddings[0]
    labels_en = labels[0]

    best_train_idx = None
    best_val_idx = None
    min_overall_val_loss = float('inf')
    results = []
    for i in range(5):
        print(f"Training model: {i}")

        # y_full = en_train_df['binary_label'] ADRC
        y_full = en_train_df['synd2'] # ADRRES-M
        train_idx, val_idx = train_test_split(
            en_train_df.index.values,
            test_size=0.2,
            stratify=y_full,
            random_state=i
        )
        en_train = torch.utils.data.TensorDataset(embeddings_en[train_idx], labels_en[train_idx])
        en_train_loader = torch.utils.data.DataLoader(en_train, batch_size=32, shuffle=True)

        input_dim = embeddings_en.shape[2]
        model = Classifier(input_dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay = 1e-2)

        result_df = train(device, model, nn.CrossEntropyLoss(),
                          optimizer, 30, en_train_loader, es_val_loader,
                          output_path, i)
        
        current_best_loss = result_df['val_loss'].iloc[0]
        if current_best_loss < min_overall_val_loss:
            min_overall_val_loss = current_best_loss
            best_train_idx = train_idx 
            best_val_idx = val_idx

        results.append(result_df)

    print(f"Pre Training Done")
    final_results_df = pd.concat(results, ignore_index=True)
    best_model = final_results_df.loc[final_results_df['val_loss'].idxmin()]
    best_model_path = best_model['checkpoint_file']
    # removing old weights
    models = final_results_df['checkpoint_file'].unique().tolist()
    for model in models:
        if model != best_model_path:
            os.remove(f"{output_path}/{model}")

    new_file_name = f"{output_path}/best_pretrain_model.pt"
    os.rename(f"{output_path}/{best_model_path}", new_file_name)
    return new_file_name, best_train_idx, best_val_idx

def get_es_tensors(es_df_subset, es_csv_df, embeddings, labels):
    """Helper to get tensors for a Spanish subset using filename matching."""
    subset_files = set(es_df_subset['file_name'])
    global_indices = es_csv_df[es_csv_df['file_name'].isin(subset_files)].index.values
    subset_X = embeddings[global_indices]
    subset_Y = labels[global_indices]
    return torch.utils.data.TensorDataset(subset_X, subset_Y)

def fine_tune(device, embeddings, labels, en_train_data, en_val_data, es_val_df, es_csv_df, best_model_path, output_path, seed):
    embeddings_en = embeddings[0]
    embeddings_es = embeddings[1]
    labels_en = labels[0]
    labels_es = labels[1]
    
    es_foldA_train, es_foldB_train, es_foldA_val, es_foldB_val = custom_balanced_2_fold(es_val_df, seed)
    es_setA = get_es_tensors(es_foldA_val, es_csv_df, embeddings_es, labels_es)
    es_setB = get_es_tensors(es_foldB_val, es_csv_df, embeddings_es, labels_es)
    
    best_models = []
    for run_num in range(1, 3):
        print(f"\n--- Starting Fine-Tuning Run {run_num} ---")
        train_es_samples = None
        val_es_samples = None
        run_output_path = None
        if run_num == 1:
            # Run 1: Train on EN + ES_A, Validate on EN + ES_B
            train_es_samples = es_setA
            val_es_samples = es_setB
            run_output_path = os.path.join(output_path, "finetune_run1")
        else: 
            # Run 2: Train on EN + ES_B, Validate on EN + ES_A
            train_es_samples = es_setB
            val_es_samples = es_setA
            run_output_path = os.path.join(output_path, "finetune_run2")
        
        os.makedirs(run_output_path, exist_ok=True)
        
        en_train_loader = torch.utils.data.DataLoader(en_train_data, batch_size=26, shuffle=True)
        en_val_loader = torch.utils.data.DataLoader(en_val_data, batch_size=26, shuffle=True)
        mixed_train_loader = MixedBatchIterator(en_train_loader, train_es_samples, batch_size=32)
        mixed_val_loader = MixedBatchIterator(en_val_loader, val_es_samples, batch_size=32)

        input_dim = embeddings_en.shape[2]
        model = Classifier(input_dim).to(device)
        model.load_state_dict(torch.load(best_model_path))
        optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay = 1e-2)
        result_df = train(
            device=device, 
            model=model, 
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=optimizer, 
            num_epoch=30, 
            train_loader=mixed_train_loader, 
            val_loader=mixed_val_loader,
            checkpoint_path=run_output_path, 
            seed=run_num
        )
        best_models.append(result_df['checkpoint_file'].iloc[0])
    
    print("\n--- Averaging Models ---")
    model1_path = os.path.join(output_path, "finetune_run1", best_models[0])
    model2_path = os.path.join(output_path, "finetune_run2", best_models[1])
    model1_state = torch.load(model1_path)
    model2_state = torch.load(model2_path)
    final_model_state = model1_state.copy()
    
    for key in final_model_state:
        final_model_state[key] = (model1_state[key] + model2_state[key]) / 2.0
    
    for run_dir in ["finetune_run1", "finetune_run2"]:
        full_path = os.path.join(output_path, run_dir)
        if os.path.exists(full_path):
            shutil.rmtree(full_path)  # removes the directory and all its contents
            print(f"Deleted: {full_path}")
    os.remove(f'{output_path}/best_pretrain_model.pt')

    return final_model_state

def split_es_val_test(es_df, seed):
    """Split Spanish data into validation (4 AD, 4 NC) and test (remaining)."""
    es_ad = es_df[es_df['synd2'] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    es_nc = es_df[es_df['synd2'] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)

    val_ad = es_ad.iloc[:4]
    val_nc = es_nc.iloc[:4]
    es_val = pd.concat([val_ad, val_nc]).sample(frac=1, random_state=seed).reset_index(drop=True)

    test_ad = es_ad.iloc[4:]
    test_nc = es_nc.iloc[4:]
    es_test = pd.concat([test_ad, test_nc]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # print(f"Values in es_val:\n{es_val}\n\nValues in es_test:\n{es_test}")
    
    return es_val, es_test

def prep_loaders(df, en_csv_df, es_csv_df, seed):
    en_train = balance_data(df[df['language'] == 'en'].copy(), seed)
    es_df = df[df['language'] == 'es'].copy()

    # CLEARED - OUTPUT CORRECT
    es_val, es_test = split_es_val_test(es_df, seed)

    # Strict inner merge on filename to preserve alignment
    def merge_split(csv_df, split_df):
        merged = csv_df.merge(split_df[['file_name', 'synd2']], on='file_name', how='inner')
        assert not merged.empty, "Merge produced empty dataframe — file names might not match!"
        return merged

    en_train_df  = merge_split(en_csv_df, en_train)
    es_val_df  = merge_split(es_csv_df, es_val)
    es_test_df = merge_split(es_csv_df, es_test)

    # print(f"es_val_df:\n{es_val_df}\n\nes_test_df:{es_test_df}")

    return en_train_df, es_val_df, es_test_df

def complete_train(device, embeddings, labels, name, seed):
    embeddings_en = embeddings[0]
    embeddings_es = embeddings[1]
    labels_en = labels[0]
    labels_es = labels[1]
    # format dataframe
    en_csv_df = pd.read_csv(f"{FEATURE_PATH}/egemapsv02_equal_n10/corresponding_files_en.csv")
    es_csv_df = pd.read_csv(f"{FEATURE_PATH}/egemapsv02_equal_n10/corresponding_files_es.csv")

    df = pd.read_excel('/home/jobe/datasets/ADReSS-M/combined_csvs.xlsx')
    en_train_df, es_val_df, es_test_df = prep_loaders(df, en_csv_df, es_csv_df, seed) # CORRECT - NO ISSUE

    # Spanish validation loader
    es_val_idx = es_csv_df.index[es_csv_df['file_name'].isin(es_val_df['file_name'])].values
    print(f"es_val_idx: {es_val_idx}")
    es_val_data = torch.utils.data.TensorDataset(embeddings_es[es_val_idx], labels_es[es_val_idx])
    es_val_loader = torch.utils.data.DataLoader(es_val_data, batch_size=32)

    # Test loader
    es_test_idx = es_csv_df.index[es_csv_df['file_name'].isin(es_test_df['file_name'])].values
    # print(f"DEBUG in complete_train: es_test_idx values: {es_test_idx}")
    torch.save(es_test_idx, os.path.join(CHECKPOINT_PATH, f"egemapsv02_equal_n10/{name}_test_idx.pt"))

    # # Phase 1: English Pretraining on Spanish validation set
    print("Beginning English Model Pretraining...")
    output_path = os.path.join(CHECKPOINT_PATH, "egemapsv02_equal_n10")
    os.makedirs(output_path, exist_ok=True)
    best_model_path, best_train_idx, best_val_idx = pre_train(
        device, 
        embeddings, 
        labels, 
        en_train_df, 
        es_val_loader, 
        output_path
    )

    # # # Phase 2: Mixed-batch transfer learning
    en_train_data = torch.utils.data.TensorDataset(embeddings_en[best_train_idx], labels_en[best_train_idx])
    en_val_data = torch.utils.data.TensorDataset(embeddings_en[best_val_idx], labels_en[best_val_idx])
    final_model = fine_tune(
        device, 
        embeddings, 
        labels, 
        en_train_data, 
        en_val_data, 
        es_val_df, 
        es_csv_df,          # look into this after i finish fixing pre-training
        best_model_path, 
        output_path,
        seed
    )
    
    final_save_path = os.path.join(output_path, f"{name}_final_averaged_model.pt")
    torch.save(final_model, final_save_path)

def custom_balanced_2_fold(df: pd.DataFrame, seed):
    #df is spanish
    nc_df = df[df['synd2']==0].reset_index(drop=True)      # ADRRES-M
    mci_ad_df = df[df['synd2']==1].reset_index(drop=True)  # ADRRES-M

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

def balance_data(df: pd.DataFrame, seed):
    # filter dataset to remove rows with no classification and fill in empty education with default value of 12
    filtered_df = df.dropna(subset=['synd2', 'age_at_recording', 'Gender']).copy()
    filtered_df = filtered_df[filtered_df['synd2'] != -1]
    filtered_df.loc[filtered_df['Education'].isna(), 'Education'] = 12
                    
    class_counts = filtered_df['synd2'].value_counts()
    print(class_counts)
    min_class_count = class_counts.min()
    
    normal_df = filtered_df[filtered_df['synd2']==0]
    mci_df = filtered_df[filtered_df['synd2']==1]

    bal_normal_df = normal_df.sample(min_class_count, random_state=seed)
    bal_mci_df = mci_df.sample(min_class_count, random_state=seed)
    full_bal_df = pd.concat([bal_normal_df, bal_mci_df], axis = 0)
    full_bal_df = full_bal_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return full_bal_df

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embeddings_en = torch.load(f"{FEATURE_PATH}/egemapsv02_equal_n10/X_en.pt")
    embeddings_es = torch.load(f"{FEATURE_PATH}/egemapsv02_equal_n10/X_es.pt")
    labels_en = torch.load(f"{FEATURE_PATH}/egemapsv02_equal_n10/Y_en.pt").long()
    labels_es = torch.load(f"{FEATURE_PATH}/egemapsv02_equal_n10/Y_es.pt").long()
    # labels = torch.where(labels==2, 1, 0)   # binary map
    embeddings = (embeddings_en, embeddings_es)
    labels = (labels_en, labels_es)
    names = ['1_adress', '2_adress', '2_adress', '3_adress', '4_adress', '5_adress'] 
    for i, name in enumerate(names):
        complete_train(device, embeddings, labels, name, i)

    
if __name__ == "__main__":
    main()
            
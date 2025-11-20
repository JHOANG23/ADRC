import os
import json
import torch
import numpy as np
import pandas as pd
from scipy import stats
from src.train import Classifier
import matplotlib.pyplot as plt
from src.config import FEATURE_PATH, EXCEL_FULL_ADRC_PATH, CHECKPOINT_PATH, METRICS_PATH
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
def eval(model_path, input_dim, test_loader, device):
    """
    Evaluate a trained AD detection model on the test set.
    Computes Accuracy, Precision, Recall, F1, AUC, and Specificity.
    """

    # --- Load model ---
    # (replace `MyModelClass` with your actual model class)
    model = Classifier(input_dim)                     # placeholder
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Run inference ---
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]  # probability of AD class (1)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # --- Compute metrics ---
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    # specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n--- Test Evaluation ---")
    print(f"Accuracy:     {acc*100:.2f}%")
    print(f"Precision:    {prec*100:.2f}%")
    print(f"Recall:       {rec*100:.2f}%")
    print(f"F1 Score:     {f1*100:.2f}%")
    # print(f"AUC:          {auc*100:.2f}%")
    # print(f"Specificity:  {spec*100:.2f}%")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        # "auc": auc,
        # "specificity": spec
    }

def eval_all(test_idx_paths, embeddings, labels, device, save_path, model_name):
    input_dim = embeddings.shape[2]
    all_metrics = []
    for file in test_idx_paths:
        test_idx = torch.load(
            os.path.join(CHECKPOINT_PATH, f"egemapsv02_equal_n10/{file}"),
            weights_only=False
        )
        print(f"test idx: {test_idx}")
        print(f"Labels: {labels[test_idx]}")
        test_data = torch.utils.data.TensorDataset(embeddings[test_idx], labels[test_idx])
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
        prefix = "_".join(file.split("_")[:2])
        model_path = os.path.join(CHECKPOINT_PATH, "egemapsv02_equal_n10", f"{prefix}_final_averaged_model.pt")

        result = eval(model_path, input_dim, test_loader, device)
        all_metrics.append(result)

    # metrics_keys = ["accuracy", "precision", "recall", "f1", "specificity"]
    metrics_keys = ["accuracy", "precision", "recall", "f1"]
    metrics_array = {key: [m[key] for m in all_metrics] for key in metrics_keys}
    mean_std_metrics = {}
    for key in metrics_keys:
        mean_std_metrics[key] = {
            "mean": float(np.mean(metrics_array[key])),
            "std": float(np.std(metrics_array[key]))
        }
    all_metrics.append({"dataset": "summary", **mean_std_metrics})
    
    json_path = f"{save_path}/eval_results.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_results = json.load(f)
        # Convert old list format to dict if necessary
        if isinstance(all_results, list):
            all_results = {"default": all_results}
    else:
        all_results = {}

    all_results[model_name] = all_metrics  # store under parent key

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nEvaluation results (including mean/std) saved to {save_path}/eval_results.json")

def cohen_d(x, y):
    """Compute Cohen's d for paired samples."""
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1)

def compare_datasets(json_path, dataset1, dataset2):
    # --- Load JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)

    # --- Extract metrics (ignore summary entries) ---
    def extract_metrics(dataset_name):
        entries = [run for run in data[dataset_name] if "dataset" not in run]
        return {
            m: [r[m] for r in entries]
            for m in ["accuracy", "precision", "recall", "f1"]
            # for m in ["accuracy", "precision", "recall", "f1", "specificity"]
        }

    metrics1 = extract_metrics(dataset1)
    metrics2 = extract_metrics(dataset2)

    # --- Compute comparison stats ---
    rows = []
    for metric in metrics1.keys():
        x = np.array(metrics1[metric])
        y = np.array(metrics2[metric])
        t_stat, p_val = stats.ttest_rel(x, y)  # paired test
        d = cohen_d(x, y)

        rows.append({
            "Metric": metric.capitalize(),
            f"{dataset1} (mean ± std)": f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}",
            f"{dataset2} (mean ± std)": f"{np.mean(y):.3f} ± {np.std(y, ddof=1):.3f}",
            "p-value": f"{p_val:.4f}",
            "Cohen's d": f"{d:.2f}"
        })

    df = pd.DataFrame(rows)
    return df


def plot_table(df, title="Statistical Comparison", save_path=None):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.7 + 1))
    ax.axis("off")
    ax.axis("tight")

    # Create the matplotlib table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.5)
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if key[0] == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embeddings = torch.load(f"{FEATURE_PATH}/egemapsv02_equal_n10/X_es.pt")
    labels     = torch.load(f"{FEATURE_PATH}/egemapsv02_equal_n10/Y_es.pt").long()
    labels     = torch.where(labels==2, 1, 0)   # binary map

    path = os.path.join(CHECKPOINT_PATH, 'egemapsv02_equal_n10')
    test_idx_paths = [f for f in os.listdir(path) if f.endswith('test_idx.pt')]

    eval_all(test_idx_paths, embeddings, labels, device, METRICS_PATH, 'ADRC')

    # json_path = '/home/jobe/ADRC/metrics/eval_results.json'
    # dataset1 = "ADRC"
    # dataset2 = "adress-m"
    # df = compare_datasets(json_path, dataset1, dataset2)
    # plot_table(df, title=f"{dataset1} vs {dataset2} Performance Comparison", save_path=f"{METRICS_PATH}/comparison_table.png")


if __name__ == '__main__':
    main()
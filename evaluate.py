import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from technical_utils import load_obj
import xml.etree.ElementTree as ET
import hydra
from omegaconf import DictConfig
from monai.transforms import Compose
import pandas as pd

# ----------------------------
# Helpers for plots & reports
# ----------------------------
def plot_confusion_matrix(cm, class_names, title="Confusion Matrix",
                          normalize=False, fname=None):
    """
    cm: 2D numpy array (square)
    class_names: list[str]
    normalize: if True, row-normalize (per true class)
    fname: optional path to save figure
    """
    cm = np.asarray(cm)
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = cm.astype(np.float64) / np.maximum(row_sums, 1e-12)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, square=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=180)
    plt.show()


def print_and_save_classification_report(y_true, y_pred, class_names,
                                         fname_txt="classification_report.txt"):
    report = classification_report(y_true, y_pred,
                                   target_names=class_names, digits=4)
    print("\nClassification Report:\n", report)
    with open(fname_txt, "w") as f:
        f.write(report)


# ----------------------------
# Your data loader (test-only)
# ----------------------------
def load_all_data(cfg, classes, micro_name, macro_name, label_name):
    """Load data from TEST section in config (as in your original evaluate)."""
    data = []
    for accession in cfg.model_data.test.accessions:
        accession_path = f"{cfg.model_data.test.path}/{accession}"
        label_path = f"{accession_path}/{label_name}.xml"
        tree = ET.parse(label_path)
        label = classes.index(tree.getroot().text)
        micro_path = f"{accession_path}/{micro_name}.npy"
        macro_path = f"{accession_path}/{macro_name}.npy"
        data.append({
            micro_name: micro_path,
            macro_name: macro_path,
            label_name: label,
            "accession": accession
        })
    return data


# ----------------------------
# Evaluation (unchanged logic)
# ----------------------------
def evaluate_and_plot(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg.model_data.classes
    micro_name = cfg.micro_macro.data.micro_name
    macro_name = cfg.micro_macro.data.macro_name
    label_name = cfg.dicom.my_keys.label_key

    all_data = load_all_data(cfg, classes, micro_name, macro_name, label_name)

    # transforms & dataset
    test_transforms = Compose([
        hydra.utils.instantiate(conf)
        for _, conf in cfg.model_data.test.transforms.items()
    ])
    DatasetCls = load_obj(cfg.model_data.test.dataset.class_name)
    dataset = DatasetCls(all_data, transform=test_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # model
    ModelCls = load_obj(cfg.model.model.class_name)
    model = ModelCls(**cfg.model.model.params).to(device)

    # load weights
    sd = torch.load(cfg.training.model_weights_dst, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    all_labels, all_preds = [], []
    TP, TN, FP, FN = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            micro = batch[micro_name].to(device)
            macro = batch[macro_name].to(device)
            label = batch[label_name].to(device).long()

            # NOTE: if your trained model expects manufacturer id, replace next line with:
            # logits = model(micro, macro, batch[cfg.dicom.my_keys.manufacturer_key].to(device).long())
            logits = model(micro, macro)

            pred = logits.argmax(1)

            y_true = label.item()
            y_pred = pred.item()
            all_labels.append(y_true)
            all_preds.append(y_pred)

            accession_id = all_data[i]["accession"]
            if y_true == 1 and y_pred == 1:
                TP.append(accession_id)
            elif y_true == 0 and y_pred == 0:
                TN.append(accession_id)
            elif y_true == 0 and y_pred == 1:
                FP.append(accession_id)
            else:  # y_true == 1 and y_pred == 0
                FN.append(accession_id)

    # ---- Confusion Matrix & plots ----
    labels_order = list(range(len(classes)))
    cm = confusion_matrix(all_labels, all_preds, labels=labels_order)
    print("Confusion Matrix:\n", cm)

    # Save both count and normalized plots
    plot_confusion_matrix(cm, classes,
                          title="Confusion Matrix (Counts)",
                          normalize=False,
                          fname="cm_counts.png")
    plot_confusion_matrix(cm, classes,
                          title="Confusion Matrix (Normalized by True Class)",
                          normalize=True,
                          fname="cm_normalized.png")

    # Classification report (print + save)
    print_and_save_classification_report(all_labels, all_preds, classes,
                                         fname_txt="classification_report.txt")

    # ---- Save accessions per outcome (as you had) ----
    results_dict = {
        "True Positives (TP)": TP,
        "False Negatives (FN)": FN,
        "True Negatives (TN)": TN,
        "False Positives (FP)": FP
    }
    out_path = "confusion_accessions.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        for key, values in results_dict.items():
            pd.DataFrame(values, columns=["accession"]).to_excel(
                writer, sheet_name=key, index=False
            )

    print("\n✅ Saved files:")
    print(" - cm_counts.png")
    print(" - cm_normalized.png")
    print(" - classification_report.txt")
    print(f" - {out_path}")


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    evaluate_and_plot(cfg)


if __name__ == "__main__":
    main()
    print("end")

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from technical_utils import load_obj
import xml.etree.ElementTree as ET
import hydra
from monai.transforms import Compose

def load_all_data(cfg, classes, micro_name, macro_name, label_name):
    """Load data from BOTH train.old, train.new and test sections in config"""
    data = []

    # # --- Train/old ---
    # for accession in cfg.model_data.train.old.accessions:
    #     accession_path = f"{cfg.model_data.train.old.path}/{accession}"
    #     label_path = f"{accession_path}/{label_name}.xml"
    #     tree = ET.parse(label_path)
    #     label = classes.index(tree.getroot().text)
    #     micro_path = f"{accession_path}/{micro_name}.npy"
    #     macro_path = f"{accession_path}/{macro_name}.npy"
    #     data.append({micro_name: micro_path, macro_name: macro_path, label_name: label})

    # # --- Train/new ---
    # for accession in cfg.model_data.train.new.accessions:
    #     accession_path = f"{cfg.model_data.train.new.path}/{accession}"
    #     label_path = f"{accession_path}/{label_name}.xml"
    #     tree = ET.parse(label_path)
    #     label = classes.index(tree.getroot().text)
    #     micro_path = f"{accession_path}/{micro_name}.npy"
    #     macro_path = f"{accession_path}/{macro_name}.npy"
    #     data.append({micro_name: micro_path, macro_name: macro_path, label_name: label})

    # --- Test ---
    for accession in cfg.model_data.test.accessions:
        accession_path = f"{cfg.model_data.test.path}/{accession}"
        label_path = f"{accession_path}/{label_name}.xml"
        tree = ET.parse(label_path)
        label = classes.index(tree.getroot().text)
        micro_path = f"{accession_path}/{micro_name}.npy"
        macro_path = f"{accession_path}/{macro_name}.npy"
        data.append({micro_name: micro_path, macro_name: macro_path, label_name: label})

    return data


# def compute_confusion_matrix(cfg):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     classes = cfg.model_data.classes
#     micro_name = cfg.micro_macro.data.micro_name
#     macro_name = cfg.micro_macro.data.macro_name
#     label_name = cfg.dicom.my_keys.label_key

#     all_data = load_all_data(cfg, classes, micro_name, macro_name, label_name)

#     # 🔹 add transforms
#     test_transforms = Compose([hydra.utils.instantiate(conf) for _, conf in cfg.model_data.test.transforms.items()])
#     dataset = load_obj(cfg.model_data.test.dataset.class_name)(all_data, transform=test_transforms)
#     loader = DataLoader(dataset, batch_size=1, shuffle=False)

#     model = load_obj(cfg.model.model.class_name)(out_channels=cfg.model.model.params.out_channels).to(device)
#     model.load_state_dict(torch.load(cfg.training.model_weights_dst))
#     model.eval()

#     all_labels, all_preds = [], []
#     with torch.no_grad():
#         for batch in loader:
#             micro = batch[micro_name].to(device)
#             macro = batch[macro_name].to(device)
#             label = batch[label_name].to(device)

#             out = model(micro, macro)
#             pred = out.argmax(1)

#             all_labels.extend(label.cpu().numpy())
#             all_preds.extend(pred.cpu().numpy())
#     # === Confusion Matrix ===
#     cm = confusion_matrix(all_labels, all_preds)
#     print("Confusion Matrix:\n", cm)
#     print("\nClassification Report:")
#     print(classification_report(all_labels, all_preds, target_names=classes))

#     # === Pretty plot ===
#     plt.figure(figsize=(6,6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=classes, yticklabels=classes)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Confusion Matrix (All Samples)")
#     plt.show()

def compute_confusion_matrix(cfg):
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg.model_data.classes
    micro_name = cfg.micro_macro.data.micro_name
    macro_name = cfg.micro_macro.data.macro_name
    label_name = cfg.dicom.my_keys.label_key

    all_data = load_all_data(cfg, classes, micro_name, macro_name, label_name)

    # 🔹 add transforms
    test_transforms = Compose([hydra.utils.instantiate(conf) for _, conf in cfg.model_data.test.transforms.items()])
    dataset = load_obj(cfg.model_data.test.dataset.class_name)(all_data, transform=test_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_obj(cfg.model.model.class_name)(out_channels=cfg.model.model.params.out_channels).to(device)
    model.load_state_dict(torch.load(cfg.training.model_weights_dst))
    model.eval()
    
    all_labels, all_preds = [], []

    # 🔹 רשימות ל־accessions
    TP, TN, FP, FN = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            micro = batch[micro_name].to(device)
            macro = batch[macro_name].to(device)
            label = batch[label_name].to(device)

            out = model(micro, macro)
            pred = out.argmax(1)

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

            # 🔹 נשלוף את ה־accession מתוך הנתיב
            accession = all_data[i]
            accession_id = accession[micro_name].split("/")[-2]  # התיקייה של ה-accession

            if label.item() == 1 and pred.item() == 1:
                TP.append(accession_id)
            elif label.item() == 0 and pred.item() == 0:
                TN.append(accession_id)
            elif label.item() == 0 and pred.item() == 1:
                FP.append(accession_id)
            elif label.item() == 1 and pred.item() == 0:
                FN.append(accession_id)

    # === Confusion Matrix ===
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # === Pretty plot ===
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (All Samples)")
    plt.show()

    # === Save Accessions by category ===
    results_dict = {
        "True Positives (TP)": TP,
        "False Negatives (FN)": FN,
        "True Negatives (TN)": TN,
        "False Positives (FP)": FP
    }

    # נשמור כקובץ Excel עם טאב לכל קטגוריה
    out_path = "confusion_accessions.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        for key, values in results_dict.items():
            df = pd.DataFrame(values, columns=["accession"])
            df.to_excel(writer, sheet_name=key, index=False)

    print(f"\n✅ Saved accessions per category to {out_path}")
    print(f"Counts: TP={len(TP)}, FN={len(FN)}, TN={len(TN)}, FP={len(FP)}")


@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    compute_confusion_matrix(cfg)

if __name__ == "__main__":
    main()
    print ('end')

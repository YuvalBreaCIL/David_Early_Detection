from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import hydra
from omegaconf import DictConfig
import xml.etree.ElementTree as ET
from monai.transforms import Compose
from technical_utils import load_obj
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
def compute_confusion_matrix(results, num_classes=2):
    """
    results: (correct, true_label, pred_label, prob_pred)
    num_classes: num of classes
    
    return confusion matrix
    [num_classes x num_classes]
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for _, true_label, pred_label, _ in results:
        cm[true_label, pred_label] += 1
    return cm

def safe_div(a, b):
    return float(a) / b if b != 0 else 0.0

def metrics_from_confusion_matrix(cm):
    """
    cm: numpy array shape (2,2)
        [[TN, FP],
         [FN, TP]]
    Return: This function calculates key classification performance metrics from a binary confusion matrix and print them.
    """
    cm = np.asarray(cm)
    assert cm.shape == (2, 2), "Confusion matrix must be 2x2: [[TN,FP],[FN,TP]]"
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

    sensitivity = recall = safe_div(TP, TP + FN)           # TPR
    specificity = safe_div(TN, TN + FP)                    # TNR
    precision  = safe_div(TP, TP + FP)                     # PPV
    npv        = safe_div(TN, TN + FN)                     # NPV
    f1         = safe_div(2 * precision * recall, precision + recall) if (precision+recall)>0 else 0.0
    accuracy   = safe_div(TP + TN, TP + TN + FP + FN)
    bal_acc    = (sensitivity + specificity) / 2.0
    fpr        = 1 - specificity
    fnr        = 1 - sensitivity

    dict= {
        "TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP),
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "sensitivity_recall_TPR": sensitivity,
        "specificity_TNR": specificity,
        "precision_PPV": precision,
        "NPV": npv,
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr,
    }
    for k,v in dict.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

# def load_all_data(cfg, classes, micro_name, macro_name, label_name, manufacturer_classes, manufacturer_name):
#     """Load ALL samples (old+new)."""
#     data = []
#     groups = [cfg.model_data.train.old, cfg.model_data.train.new, cfg.model_data.test]
#     for group in groups:
#         for accession in group.accessions:
#             accession_path = f"{group.path}/{accession}"
#             label_path = f"{accession_path}/{label_name}.xml"
#             tree = ET.parse(label_path)
#             label = classes.index(tree.getroot().text)
#             micro_path = f"{accession_path}/{micro_name}.npy"
#             macro_path = f"{accession_path}/{macro_name}.npy"
            
#             # root = tree.getroot()
#             # machine_type = root.findtext("metadata/machine_type")
#             # manufacturer_idx = manufacturer_classes.index(machine_type)


#             manufacturer_path = f"{accession_path}/{manufacturer_name}.xml"
#             tree_machine_type = ET.parse(manufacturer_path)
#             manufacturer = manufacturer_classes.index(tree_machine_type.getroot().text)

#             data.append({micro_name: micro_path, macro_name: macro_path, label_name: label, manufacturer_name: manufacturer})
#     return data
def load_all_data(cfg, classes, micro_name, macro_name, label_name,
                  manufacturer_classes, manufacturer_name):
    """
    Load ALL samples (train.old + train.new + test), attach label and manufacturer index.

    Expects, per accession directory:
      - {label_name}.xml     -> contains just the label text (as you already save)
      - {micro_name}.npy
      - {macro_name}.npy
      - {manufacturer_name}.xml  -> contains manufacturer text (e.g., "GE MEDICAL SYSTEMS", "Siemens")

    If manufacturer is missing or not in `manufacturer_classes`, the sample is skipped
    (with a warning), to avoid crashing later when converting to indices.
    """
    data = []
    groups = [cfg.model_data.train.old, cfg.model_data.train.new, cfg.model_data.test]

    # Normalize manufacturer class list once (case-insensitive match)
    manu_norm_list = [m.strip().upper() for m in manufacturer_classes]

    for group in groups:
        base_path = group.path
        for accession in group.accessions:
            accession_path = f"{base_path}/{accession}"

            # paths
            label_path = f"{accession_path}/{label_name}.xml"
            micro_path = f"{accession_path}/{micro_name}.npy"
            macro_path = f"{accession_path}/{macro_name}.npy"
            manu_path  = f"{accession_path}/{manufacturer_name}.xml"

            # ---- label ----
            try:
                tree = ET.parse(label_path)
                label_txt = tree.getroot().text
                label_idx = classes.index(label_txt)
            except Exception as e:
                print(f"⚠️ Skipping {accession}: failed to read label from {label_path} ({e})")
                continue

            # ---- manufacturer (preferred: dedicated XML) ----
            if not os.path.exists(manu_path):
                print(f"⚠️ Skipping {accession}: manufacturer file missing at {manu_path}")
                continue

            try:
                manu_tree = ET.parse(manu_path)
                manu_txt = (manu_tree.getroot().text or "").strip().upper()
                manu_idx = manu_norm_list.index(manu_txt)
            except ValueError:
                print(f"⚠️ Skipping {accession}: unknown manufacturer '{manu_txt}' "
                      f"(expected one of {manufacturer_classes})")
                continue
            except Exception as e:
                print(f"⚠️ Skipping {accession}: failed to read manufacturer from {manu_path} ({e})")
                continue

            # ---- add sample ----
            data.append({
                micro_name: micro_path,
                macro_name: macro_path,
                label_name: label_idx,
                manufacturer_name: manu_idx,
                "accession": accession,
            })

    return data

def load_train_data(cfg, classes, micro_name, macro_name, label_name,
                  manufacturer_classes, manufacturer_name):
    """
    Load ALL samples (train.old + train.new + test), attach label and manufacturer index.

    Expects, per accession directory:
      - {label_name}.xml     -> contains just the label text (as you already save)
      - {micro_name}.npy
      - {macro_name}.npy
      - {manufacturer_name}.xml  -> contains manufacturer text (e.g., "GE MEDICAL SYSTEMS", "Siemens")

    If manufacturer is missing or not in `manufacturer_classes`, the sample is skipped
    (with a warning), to avoid crashing later when converting to indices.
    """
    data = []
    groups = [cfg.model_data.train.old, cfg.model_data.train.new]

    # Normalize manufacturer class list once (case-insensitive match)
    manu_norm_list = [m.strip().upper() for m in manufacturer_classes]

    for group in groups:
        base_path = group.path
        for accession in group.accessions:
            accession_path = f"{base_path}/{accession}"

            # paths
            label_path = f"{accession_path}/{label_name}.xml"
            micro_path = f"{accession_path}/{micro_name}.npy"
            macro_path = f"{accession_path}/{macro_name}.npy"
            manu_path  = f"{accession_path}/{manufacturer_name}.xml"

            # ---- label ----
            try:
                tree = ET.parse(label_path)
                label_txt = tree.getroot().text
                label_idx = classes.index(label_txt)
            except Exception as e:
                print(f"⚠️ Skipping {accession}: failed to read label from {label_path} ({e})")
                continue

            # ---- manufacturer (preferred: dedicated XML) ----
            if not os.path.exists(manu_path):
                print(f"⚠️ Skipping {accession}: manufacturer file missing at {manu_path}")
                continue

            try:
                manu_tree = ET.parse(manu_path)
                manu_txt = (manu_tree.getroot().text or "").strip().upper()
                manu_idx = manu_norm_list.index(manu_txt)
            except ValueError:
                print(f"⚠️ Skipping {accession}: unknown manufacturer '{manu_txt}' "
                      f"(expected one of {manufacturer_classes})")
                continue
            except Exception as e:
                print(f"⚠️ Skipping {accession}: failed to read manufacturer from {manu_path} ({e})")
                continue

            # ---- add sample ----
            data.append({
                micro_name: micro_path,
                macro_name: macro_path,
                label_name: label_idx,
                manufacturer_name: manu_idx,
                "accession": accession,
            })

    return data

def train_one_fold(cfg, train_dataset, test_dataset, train_idx, test_idx, fold_num):
    """Train on train_idx and evaluate on test_idx (LOO one sample)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg.model_data.classes
    micro_name = cfg.micro_macro.data.micro_name
    macro_name = cfg.micro_macro.data.macro_name
    label_name = cfg.dicom.my_keys.label_key

    # subsets
    train_subset = Subset(train_dataset, train_idx)
    test_subset = Subset(test_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=cfg.model_data.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=1)

    # model + optimizer
    model = load_obj(cfg.model.model.class_name)(out_channels=cfg.model.model.params.out_channels).to(device)
    optimizer = load_obj(cfg.optimizer.class_name)(model.parameters(), cfg.optimizer.learning_rate)

    # simple training loop
    # index=1
    for epoch in range(cfg.training.num_epochs):
        model.train()
        for batch_data in train_loader:
            micro = batch_data[micro_name].to(device)
            macro = batch_data[macro_name].to(device)
            label = batch_data[label_name].to(device)
            manu_id = batch_data[cfg.dicom.my_keys.manufacturer_key].to(device)  # LongTensor
            manu_id = manu_id.long()
            # print(
            #     f"[SANITY][fold {fold_num}][train][epoch {epoch+1}] "
            #     f"micro shape={tuple(micro.shape)} macro shape={tuple(macro.shape)}"
            # )


            optimizer.zero_grad()
            # added to include the manufacturer
            # num_manu = len(cfg.model_data.manufacturer)  # למשל ["GE MEDICAL SYSTEMS", "Siemens"]
            # out =model(out_channels=cfg.model.model.params.out_channels,
            #             num_manufacturers=num_manu).to(device)
            # If I want to give weights to machine
              # 1. איבוד פר־דגימה (לא ממוצע עדיין!)
            # loss_per_sample = F.cross_entropy(out, label, reduction='none')

            # # 2. פקטור משקל לפי מכונה
            # # נניח:
            # # cfg.model_data.manufacturer = ["GE MEDICAL SYSTEMS", "SIEMENS"]
            # manufacturer_classes = cfg.model_data.manufacturer
            # # למצוא את האינדקס של Siemens (בצורה לא רגישה לאותיות)
            # siemens_idx = next(
            #     i for i, name in enumerate(manufacturer_classes)
            #     if "SIEMENS" in name.upper()
            # )

            # siemens_factor = 2.0  # כמה "יותר חשוב" Siemens מג׳י-אי
            # manu_factor = torch.where(
            #     manu_id == siemens_idx,
            #     torch.tensor(siemens_factor, device=device),
            #     torch.tensor(1.0, device=device)
            # )

            # # 3. ממוצע משוקלל
            # loss = (loss_per_sample * manu_factor).mean()

            # loss.backward()
            # optimizer.step()
            
            out = model(micro, macro,manu_id)
            loss = F.cross_entropy(out, label)
            loss.backward()
            optimizer.step()
        # print (f"epoch number {index}")
        # index+=1

    # evaluate on the one test sample
    model.eval()
    # with torch.no_grad():
    for batch_data in test_loader:
        micro = batch_data[micro_name].to(device)
        macro = batch_data[macro_name].to(device)
        label = batch_data[label_name].to(device)
        # pred = model(micro, macro).argmax(1)
        # logits = model(micro, macro)
        manu_id = batch_data[cfg.dicom.my_keys.manufacturer_key].to(device).long()
        # print(
        #     f"[SANITY][fold {fold_num}][test] "
        #     f"micro shape={tuple(micro.shape)} macro shape={tuple(macro.shape)}"
        # )
        logits = model(micro, macro, manu_id)
        probs = torch.softmax(logits, dim=1)   # הסתברויות לכל מחלקה
        pred  = torch.argmax(probs, dim=1)     # חיזוי סופי
        
        # הסתברות של המחלקה החזויה (batch size = 1 אז לוקחים אינדקס 0)
        prob_pred = probs[0, pred.item()].item()
        #Compute gradcam for macro image
        compute_gradcam_macro(model,macro,micro,manu_id,fold_num,batch_data,label,pred,prob_pred)
        # grab manufacturer id of the test sample
        machine_type = batch_data[cfg.dicom.my_keys.manufacturer_key].item()

        correct = int((pred == label).sum().item())
        return correct, label.item(), pred.item(), prob_pred, machine_type

from sklearn.model_selection import LeaveOneOut
import numpy as np
import hydra
from monai.transforms import Compose
from technical_utils import load_obj

def leave_one_out_cv(cfg):
    """Run Leave-One-Out Cross Validation."""
    # שליפת שמות ותוויות מהקונפיג
    classes = cfg.model_data.classes
    manufacturer_classes = cfg.model_data.manufacturer
    micro_name = cfg.micro_macro.data.micro_name
    macro_name = cfg.micro_macro.data.macro_name
    label_name = cfg.dicom.my_keys.label_key
    manufacturer_name=  cfg.dicom.my_keys.manufacturer_key
    

    # load all data (train + test, old + new)
    all_data = load_all_data(cfg, classes, micro_name, macro_name, label_name, manufacturer_classes, manufacturer_name)
    labels = [d[label_name] for d in all_data]
    accessions=[d["accession"] for d in all_data]
    num_benign = sum([l == 0 for l in labels])
    num_tumor = sum([l == 1 for l in labels])

    # accession -> data group mapping for export
    def _acc_set(group_name):
        group_cfg = cfg.dicom.data.get(group_name, None)
        if group_cfg is None or group_cfg.accessions is None:
            return set()
        return {str(a) for a in group_cfg.accessions}

    group_order = [
        "tumor_new", "tumor_subcm", "tumor_old",
        "benign_subcm", "benign_new", "benign_old",
    ]
    group_to_accessions = {g: _acc_set(g) for g in group_order}

    def get_data_group(accession):
        acc = str(accession)
        for g in group_order:
            if acc in group_to_accessions[g]:
                return g
        return "unknown"

    print(f"Total samples: {len(labels)}")
    print(f"Benign: {num_benign}, Tumor: {num_tumor}")
    print(f"Baseline accuracy (always predict majority): {max(num_benign, num_tumor) / len(labels):.4f}")

    # ---------- define base transforms ----------
    base_transform_list = [
        hydra.utils.instantiate(conf)
        for _, conf in cfg.model_data.train.transforms.items()
    ]

    # ---------- define augmentation transforms ----------
    aug_transform_list = []
    if "augmentation" in cfg.model_data.train and cfg.model_data.train.augmentation is not None:
        print(f"cfg.model_data.train.augmentation:{cfg.model_data.train.augmentation}")
        aug_transform_list = [
            hydra.utils.instantiate(conf)
            for _, conf in cfg.model_data.train.augmentation.items()
        ]

    test_transforms = Compose(base_transform_list)
    train_transforms = Compose(base_transform_list + aug_transform_list)

    # create both datasets from all_data so LOO indices remain aligned
    dataset_class = load_obj(cfg.model_data.train.dataset.class_name)
    test_dataset = dataset_class(all_data, transform=test_transforms)
    train_dataset = dataset_class(all_data, transform=train_transforms)
    # הגדרת Leave-One-Out
    loo = LeaveOneOut()
    results = []

    # מעבר על כל פיצול LOO
    for fold_num, (train_idx, test_idx) in enumerate(loo.split(np.zeros(len(labels)), labels)):
        # train_idx -> רשימת אינדקסים לאימון
        # test_idx  -> רשימת אינדקס יחיד לבדיקה
        correct, true_label, pred_label, prob_pred, machine_type = train_one_fold(
            cfg, train_dataset, test_dataset, train_idx, test_idx, fold_num
        )
        accession = accessions[test_idx[0]]  # <-- ה-accession האמיתי מה-all_data
        results.append((correct, true_label, pred_label, prob_pred, machine_type, accession))

        print(f"Fold {fold_num+1}/{len(labels)}- Accession: {accession} - "
              f"True: {true_label}, Pred: {pred_label}, with prob of {prob_pred:.3f} Correct: {correct}")

    # חישוב דיוק ממוצע
    acc = np.mean([r[0] for r in results])
    print(f"\nLeave-One-Out CV Accuracy: {acc:.4f}")
    cm = compute_confusion_matrix(results, num_classes=len(classes))
    print("\nConfusion Matrix:")
    print(cm)
    metrics_from_confusion_matrix(cm)
    
    #-- Addition for export excel file --#
    # --- Per-manufacturer confusion matrices & metrics ---
    per_manu = compute_confusion_by_manufacturer(results, manufacturer_classes)
    for manu_name, cm_manu in per_manu.items():
        print(f"\nManufacturer: {manu_name}")
        print(cm_manu)
        metrics_from_confusion_matrix(cm_manu)
    
    rows = []
    for (correct, t, p, prob, manu_id, accession) in results:
        # prepare a row with zeros for all manufacturers*{TP,TN,FP,FN}
        row = {
            "accession": accession,
            "data_group": get_data_group(accession),
            "manufacturer": manu_id,
            "true_label": t,
            "pred_label": p,
            "pred_prob_of_pred_class": prob,
            "is_correct": int(correct),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Save to Excel (Hydra changes workdir; this will land in the Hydra run dir)
    out_xlsx = os.path.join(os.getcwd(), "per_accession_by_manufacturer.xlsx")
    df.to_excel(out_xlsx, index=False)
    print(f"\nSaved per-accession manufacturer confusion table to: {out_xlsx}")
    
    return results


def compute_confusion_matrix(results, num_classes=2):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for _, t, p, *_ in results:
        cm[t, p] += 1
    return cm

def compute_confusion_by_manufacturer(results, manufacturer_classes):
    # split results by manu_id
    out = {}
    for manu_id in range(len(manufacturer_classes)):
        # subset = [r for r in results if r[-1] == manu_id]
        subset = [r for r in results if r[-2] == manu_id]  # r = (.., manu_id, accession) -> manu_id הוא באינדקס -2
        cm = compute_confusion_matrix(subset, num_classes=2)
        out[manufacturer_classes[manu_id]] = cm
    return out

# === GradCam functions ===
def compute_gradcam_macro(model,macro,micro,manu_id,fold_num,batch_data,label,pred,prob_pred):
    # target_layer = model.features_macro[9]  # conv האחרון לפני ה-pool האחרון
    target_layer = model.features_macro[9]  # זה ה-ReLU אחרי ה-MaxPool האחרון

    cam, pred_class, probs2 = gradcam_2d_on_macro(
        model=model,
        macro=macro,
        micro=micro,          # עדיין צריך להעביר כי forward דורש
        manu_id=manu_id,
        target_layer=target_layer,
        class_idx=None        # לפי המחלקה החזויה
    )

    out_dir ="/mnt/breacil/Yuval/Early detection/try_for_david_data/gradcam_mcaro_after_relu" #"/media/breacil/Yuval/Early detection/try_for_david_data/gradcam_macro" #os.path.join(os.getcwd(), "gradcam_macro")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"fold_{fold_num:03d}_acc_{batch_data['accession'][0] if 'accession' in batch_data else 'NA'}.png")

    save_gradcam_png_macro(
        macro=macro,
        cam=cam,
        out_path=out_png,
        title=f"MACRO Grad-CAM | True={int(label.item())} Pred={int(pred.item())} P={prob_pred:.3f}"
    )
def gradcam_2d_on_macro(model, macro, micro, manu_id, target_layer, class_idx=None):
    """
    עושה Grad-CAM על ה-MACRO בלבד (כלומר hook על features_macro),
    אבל עדיין מריץ את המודל עם כל הקלטים (micro, macro, manu_id) כי forward שלך דורש אותם.
    
    macro: [B,1,H,W]
    returns:
      cam_up: [B,1,H,W] מנורמל ל-[0,1]
      pred_class: int
      probs: [B,num_classes]
    """
    model.eval()

    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out)      # [B,C,h,w]

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])  # [B,C,h,w]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # צריך grads
    macro = macro.requires_grad_(True)

    logits = model(micro, macro, manu_id)  # המודל שלך דורש micro+macro+manu
    probs = torch.softmax(logits, dim=1)

    pred_class = int(torch.argmax(probs, dim=1).item())
    if class_idx is None:
        class_idx = pred_class

    score = logits[:, class_idx].sum()

    model.zero_grad(set_to_none=True)
    score.backward()

    h1.remove()
    h2.remove()

    acts = activations[0]   # [B,C,h,w]
    grads = gradients[0]    # [B,C,h,w]

    # weights: GAP על h,w
    weights = grads.mean(dim=(2,3), keepdim=True)       # [B,C,1,1]
    cam = (weights * acts).sum(dim=1, keepdim=True)     # [B,1,h,w]
    cam = F.relu(cam)

    # upsample לגודל macro המקורי
    cam_up = F.interpolate(cam, size=macro.shape[-2:], mode="bilinear", align_corners=False)

    # normalize ל-[0,1]
    cam_up = cam_up - cam_up.min()
    cam_up = cam_up / (cam_up.max() + 1e-8)

    return cam_up, pred_class, probs

import matplotlib.pyplot as plt
def save_gradcam_png_macro(macro, cam, out_path, title="Grad-CAM (macro)"):
    """
    macro: [1,1,H,W]
    cam:   [1,1,H,W]
    """
    img = macro[0, 0].detach().cpu()
    hm  = cam[0, 0].detach().cpu()

    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.imshow(hm, alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    leave_one_out_cv(cfg)

if __name__ == "__main__":
    main()

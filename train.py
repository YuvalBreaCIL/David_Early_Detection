import torch
from torch.utils.data import DataLoader
from monai.transforms import (Compose)
import os
import hydra
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from technical_utils import load_obj
import torch.nn.functional as F

def load_data(cfg, classes, micro_name, macro_name, label_name):
    """
    build this function yourself if you want, notice the data format as a list of dictionaries with micro, macro and label
    as keys
    :param dir_path:
    :param accessions: the accessions to load, if you want to use k-fold just train multiple times each time changing
    the accessions to load for train and test.
    :return: data in format of list of dictionaries {micro_key: micro, macro_key: macro, label_key: label}
    """
    data = []
    groups = [
        cfg.model_data.train.old,
        cfg.model_data.train.new,

    ]

    for group in groups:
        for accession in group.accessions:
            accession_path = f"{group.path}/{accession}"
            label_path = f"{accession_path}/{label_name}.xml"

            tree = ET.parse(label_path)
            label = classes.index(tree.getroot().text)
            micro_path = f"{accession_path}/{micro_name}.npy"
            macro_path = f"{accession_path}/{macro_name}.npy"
            data.append({micro_name: micro_path, macro_name: macro_path, label_name: label})
    return data

def train(cfg):
    log_path = f"{cfg.log.dir_path}/{cfg.log.name}"
    os.makedirs(log_path,exist_ok=True)# mkdir(log_path)
    writer = SummaryWriter(log_path)
    print(log_path)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    classes = cfg.model_data.classes
    micro_name = cfg.micro_macro.data.micro_name
    macro_name= cfg.micro_macro.data.macro_name
    label_name = cfg.dicom.my_keys.label_key

    train_data = load_data(cfg, classes, micro_name, macro_name, label_name)
    test_data = load_data(cfg, classes, micro_name, macro_name, label_name)

    num_benign = sum([d[label_name] == 0 for d in train_data])
    num_tumors = sum([d[label_name] == 1 for d in train_data])
    weights = torch.tensor([num_tumors, num_benign], device=device).float()

    train_transforms = Compose([hydra.utils.instantiate(conf) for _, conf in cfg.model_data.train.transforms.items()])
    test_transforms = Compose([hydra.utils.instantiate(conf) for _, conf in cfg.model_data.test.transforms.items()])

    train_dataset = load_obj(cfg.model_data.train.dataset.class_name)(train_data, transform=train_transforms)
    test_dataset = load_obj(cfg.model_data.test.dataset.class_name)(test_data, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=cfg.model_data.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.model_data.test.batch_size)

    model = load_obj(cfg.model.model.class_name)(out_channels=cfg.model.model.params.out_channels).to(device)
    if cfg.model.pretrained_path:
        model.load_state_dict(torch.load(cfg.training.pretrained_path))
    optimizer = load_obj(cfg.optimizer.class_name)(model.parameters(), cfg.optimizer.learning_rate)

    # # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    train_losses = []
    val_metrics = []
    for epoch_number in range(cfg.training.num_epochs):
        # TRAIN

        print(f"epoch {epoch_number + 1}/{cfg.training.num_epochs}")
        model.train()
        epoch_train_loss = 0
        
        for step, batch_data in tqdm(enumerate(train_loader)):
            micro = batch_data[micro_name].to(device)
            macro = batch_data[macro_name].to(device)
            label = batch_data[label_name].to(device)
            optimizer.zero_grad()
            output = model(micro, macro)
            loss = F.cross_entropy(output, label, weight=weights)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()


        epoch_train_loss /= (step + 1)
        train_losses.append(epoch_train_loss)
        writer.add_scalar(f"train_loss", epoch_train_loss, global_step=epoch_number+1)
        print(f"epoch {epoch_number + 1} average loss: {epoch_train_loss:.4f}")


        # EVAL
        if (epoch_number+1) % cfg.log.val_interval == 0:
            model.eval()
            num_correct = 0
            with torch.no_grad():
                for batch_data in tqdm(test_loader):
                    micro = batch_data[micro_name].to(device)
                    macro = batch_data[macro_name].to(device)
                    label = batch_data[label_name].to(device)

                    output = model(micro, macro)
                    output = output.argmax(1)
                    num_correct += torch.sum(output==label)

        val_metric = num_correct / len(test_dataset)
        if val_metric > best_metric:
            best_metric = val_metric
            best_metric_epoch = epoch_number + 1
            torch.save(model.state_dict(), cfg.training.model_weights_dst)
            print("saved new best metric model")
        print(
            "current epoch: {} current val accuracy dice: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                epoch_number + 1, val_metric, best_metric, best_metric_epoch
            )
        )
        val_metrics.append(val_metric)
        print(f"val accuracy: [{num_correct}/{len(test_dataset)}]")
        writer.add_scalar(f"val_metric", val_metric, global_step=epoch_number + 1)

   
    # results_table_np[1, :len(val_metrics)] = val_metrics
    # === החל מכאן: יצירת טבלת תוצאות בצורה עמידה ===
    # אם train_losses ו-val_metrics באורכים שונים, ניקח את המקסימום
    n_cols = max(len(train_losses), len(val_metrics))
    results_table_np = np.zeros((2, n_cols), dtype=np.float32)

    # שורה 0: הפסדים באימון
    results_table_np[0, :len(train_losses)] = np.asarray(train_losses, dtype=np.float32)

    # שורה 1: ולידציה – המרה בטוחה גם אם יש טנזורים
    vals = [
        float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
        for v in val_metrics
    ]
    results_table_np[1, :len(vals)] = np.asarray(vals, dtype=np.float32)

    df = pd.DataFrame(data=results_table_np, index=['train_losses', 'val_metrics'], columns=[f"epoch {i+1}" for i in range(epoch_number+1)])
    df.to_excel(f"{log_path}/results.xlsx")
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()

    print('end')

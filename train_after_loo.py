# import torch, xml.etree.ElementTree as ET, os
# from torch.utils.data import DataLoader
# from monai.transforms import Compose
# import hydra
# from omegaconf import DictConfig
# import torch.nn.functional as F
# from technical_utils import load_obj

# def load_train_only(cfg, classes, micro_name, macro_name, label_name):
#     data = []
#     for group in [cfg.model_data.train.old, cfg.model_data.train.new]:
#         for accession in group.accessions:
#             p = f"{group.path}/{accession}"
#             y = classes.index(ET.parse(f"{p}/{label_name}.xml").getroot().text)
#             data.append({micro_name:f"{p}/{micro_name}.npy",
#                          macro_name:f"{p}/{macro_name}.npy",
#                          label_name:y})
#     return data

# def train_full_and_save(cfg):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     classes = cfg.model_data.classes
#     micro_name = cfg.micro_macro.data.micro_name
#     macro_name = cfg.micro_macro.data.macro_name
#     label_name = cfg.dicom.my_keys.label_key

#     train_data = load_train_only(cfg, classes, micro_name, macro_name, label_name)

#     transforms = Compose([hydra.utils.instantiate(c) for _, c in cfg.model_data.train.transforms.items()])
#     dataset = load_obj(cfg.model_data.train.dataset.class_name)(train_data, transform=transforms)
#     loader = DataLoader(dataset, batch_size=cfg.model_data.train.batch_size, shuffle=True)

#     model = load_obj(cfg.model.model.class_name)(out_channels=cfg.model.model.params.out_channels).to(device)
#     optimizer = load_obj(cfg.optimizer.class_name)(model.parameters(), cfg.optimizer.learning_rate)

#     for epoch in range(cfg.training.num_epochs):
#         model.train()
#         for batch in loader:
#             micro = batch[micro_name].to(device)
#             macro = batch[macro_name].to(device)
#             label = batch[label_name].to(device)
#             optimizer.zero_grad()
#             out = model(micro, macro)
#             loss = F.cross_entropy(out, label)
#             loss.backward()
#             optimizer.step()

#     # שמירת המשקולות למיקום שהוגדר בקונפיג
#     os.makedirs(os.path.dirname(cfg.training.model_weights_dst), exist_ok=True)
#     torch.save(model.state_dict(), cfg.training.model_weights_dst)
#     print(f"Saved model to: {cfg.training.model_weights_dst}")

# @hydra.main(config_path='conf', config_name='config')
# def main(cfg: DictConfig):
#     train_full_and_save(cfg)

# if __name__ == "__main__":
#     main()

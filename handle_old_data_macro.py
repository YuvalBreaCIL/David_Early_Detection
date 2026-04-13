import os
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
## The old data macro file was not in the size of the macro i defined, so this code change all the macro size to be according to the fefined size in the yaml file (currently 50,50)
@hydra.main(version_base="1.3", config_path='conf', config_name='config')
def main(cfg: DictConfig):
    # base_path = "/media/breacil/Yuval/Early detection/try_for_david_data/model_data"
    base_path= cfg.micro_macro.data.path
    model_shape=cfg.micro_macro.array_info.macro_shape
    half_new=cfg.micro_macro.array_info.macro_shape[0] //2

    base_path = cfg.micro_macro.data.path

    model_shape = tuple(cfg.micro_macro.array_info.macro_shape)  # (50, 50)
    half_new = model_shape[0] // 2  # 25

    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        macro_path = os.path.join(folder_path, "macro.npy")

        if not os.path.exists(macro_path):
            print(f"{folder_name}: macro.npy not found")
            continue

        try:
            macro = np.load(macro_path)
            print(f"{folder_name}: macro shape = {macro.shape}")

            if macro.shape != model_shape:
                h, w = macro.shape
                center_y = h // 2
                center_x = w // 2

                y_min = center_y - half_new
                y_max = center_y + half_new
                x_min = center_x - half_new
                x_max = center_x + half_new

                macro_cropped = macro[y_min:y_max, x_min:x_max]

                if macro_cropped.shape != model_shape:
                    raise RuntimeError(
                        f"{folder_name}: crop failed, got {macro_cropped.shape}"
                    )

                np.save(macro_path, macro_cropped)
                print(f"{folder_name}: cropped Macro.npy from {macro.shape} → {model_shape}")

            else:
                print(f"{folder_name}: skipped (shape={macro.shape})")

        except Exception as e:
            print(f"{folder_name}: error loading macro.npy ({e})")

if __name__=="__main__":
    main()
    print('end')
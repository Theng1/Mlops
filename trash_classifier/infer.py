import os
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils import evaluate
from pathlib import Path

class TrashDataset(Dataset):
    def __init__(self, folder_contents, transform, class_to_idx):
        self.folder_contents = folder_contents
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.folder_contents)

    def __getitem__(self, idx):
        image_path = self.folder_contents[idx]
        image = Image.open(image_path).convert("RGB")
        label_name = Path(image_path).parent.name
        label = self.class_to_idx[label_name]
        return self.transform(image).float() / 255, torch.tensor(label), image_path.split("/")[-1]

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    val_dir = cfg["infer"]["path_to_val"]
    all_images = []
    for class_name in os.listdir(val_dir):
        class_dir = os.path.join(val_dir, class_name)
        all_images += [os.path.join(class_dir, f) for f in os.listdir(class_dir)]

    class_to_idx = {name: idx for idx, name in enumerate(sorted(os.listdir(val_dir)))}

    dataset = TrashDataset(all_images, transform, class_to_idx)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.load(cfg["infer"]["model_path"])
    model = torch.load(cfg["infer"]["model_path"], weights_only=False)
    model = model.to(device)

    names, preds = [], []
    acc = evaluate(model, loader, names, preds, device) * 100

    val_df = pd.DataFrame({"Image Name": names, "Prediction": preds})
    val_df = val_df.reset_index(drop=True)
    val_df.to_csv(cfg["infer"]["result_save"])
    print(f"Accuracy: {acc.round()}%")

    # pd.DataFrame({"Image": names, "Predicted Class": preds}).to_csv(cfg["infer"]["result_save"], index=False)
    # print(f"Validation accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()

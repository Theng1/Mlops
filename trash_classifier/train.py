import os
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Adam
from torchvision import transforms, models
from torch.utils.data import DataLoader
from utils import TrashDataset, train_one_epoch

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    train_dir = cfg["train"]["path_to_train"]
    all_images = []
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        all_images += [os.path.join(class_dir, f) for f in os.listdir(class_dir)]

    class_to_idx = {name: idx for idx, name in enumerate(sorted(os.listdir(train_dir)))}

    train_dataset = TrashDataset(all_images, transform, class_to_idx)
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    pretrained_resnet = models.resnet18(weights=cfg["resnet"]["weights"])
    pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, cfg["resnet"]["num_classes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pretrained_resnet.to(device)

    optimizer = Adam(model.parameters(), lr=cfg["resnet"]["lr"])
    loss = nn.CrossEntropyLoss()

    for epoch in range(cfg["resnet"]["no_epochs"]):
        loss_value = train_one_epoch(model, dataloader, loss, optimizer, device)
        print(f"Epoch {epoch+1} loss: {loss_value:.4f}")

    torch.save(model, cfg["train"]["model_save"])

if __name__ == "__main__":
    main()

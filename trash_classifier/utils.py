import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm.auto import tqdm
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
        return self.transform(image).float() / 255, torch.tensor(label, dtype=torch.long)

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for image, label in tqdm(dataloader):
        image, label = image.to(device), label.to(device)
        preds = model(image)
        loss = loss_fn(preds, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, image_names, image_preds, device):
    model.eval()
    total_correct = 0
    total = 0
    for image, label, img_name in tqdm(dataloader):
        image, label = image.to(device), label.to(device)
        preds = model(image).argmax(dim=1)
        correct = (preds == label).sum()
        total += preds.shape[0]
        total_correct += correct
        image_names.extend(list(img_name))
        image_preds.extend(preds.cpu().numpy().astype(int))
    return total_correct / total

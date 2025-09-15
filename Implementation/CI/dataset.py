import torchvision.transforms as T
from torch.utils.data import Dataset

class FEMNISTDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"]
        label = int(item["character"])
        if self.transform:
            img = self.transform(img)
        return img, label

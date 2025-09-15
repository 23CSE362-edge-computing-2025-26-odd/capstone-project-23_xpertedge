import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from datasets import load_dataset
from tqdm import tqdm
from dataset import FEMNISTDataset

print("Loading FEMNIST dataset from HuggingFace...")
ds = load_dataset("flwrlabs/femnist")

transform = T.Compose([
    T.Resize((28,28)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

temp_split = ds['train'].train_test_split(test_size=0.1, seed=42)
test_split = temp_split['test']

train_valid = temp_split['train'].train_test_split(test_size=0.1, seed=42)
train_ds = FEMNISTDataset(train_valid['train'], transform)
val_ds   = FEMNISTDataset(train_valid['test'], transform)
test_ds  = FEMNISTDataset(test_split, transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=62, width_mult=1.0).to(device)

print("Model parameters:", sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=2e-4)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(epoch):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}"):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * xb.size(0)
        total += xb.size(0)
        correct += (pred.argmax(1) == yb).sum().item()
    return loss_sum/total, correct/total

def evaluate():
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss_sum += loss.item() * xb.size(0)
            total += xb.size(0)
            correct += (pred.argmax(1) == yb).sum().item()
    return loss_sum/total, correct/total

def test_accuracy():
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            correct += (pred.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total

for epoch in range(1, 6):
    tr_loss, tr_acc = train_one_epoch(epoch)
    val_loss, val_acc = evaluate()
    print(f"Epoch {epoch}: train_acc={tr_acc:.3f}, val_acc={val_acc:.3f}")

torch.save(model.state_dict(), "tiny_femnist.pth")
print("Model saved as tiny_femnist.pth")

model = CNN(num_classes=62, width_mult=1.0).to(device)
model.load_state_dict(torch.load("tiny_femnist.pth", map_location=device))
model.eval()

val_loss, val_acc = evaluate()
print(f"Validation accuracy: {val_acc*100:.2f}%")

model.load_state_dict(torch.load("tiny_femnist.pth", map_location=device))

final_test_acc = test_accuracy()
print(f"Test Accuracy on Unseen Data: {final_test_acc*100:.2f}%")
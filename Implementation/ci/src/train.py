import torch
import torch.nn as nn
from .models import CNN

def train_and_evaluate_model(hyperparams, train_loader, val_loader, device, max_epochs=3):
    try:
        model = CNN(num_classes=62, hyperparams=hyperparams).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hyperparams.learning_rate,
            momentum=hyperparams.momentum,
            weight_decay=hyperparams.weight_decay
        )

        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        best_val_acc = 0.0
        
        for epoch in range(max_epochs):
            model.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx > 200:
                    break
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    if batch_idx > 50:
                        break
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)

            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)
            scheduler.step()
        return best_val_acc
    
    except Exception as e:
        print(f"Error training model: {e}")
        return 0.0

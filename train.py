import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import random_split

def graph_loader(graph_dataset):
    train_len = int(0.8 * len(graph_dataset))
    val_len = int(0.1 * len(graph_dataset))
    test_len = len(graph_dataset) - train_len - val_len

    train_ds, val_ds, test_ds = random_split(graph_dataset, [train_len, val_len, test_len])

    return (
        GeometricDataLoader(train_ds, shuffle=True),
        GeometricDataLoader(val_ds, shuffle=False),
        GeometricDataLoader(test_ds, shuffle=False)
    )

def train(model, device, train_loader, val_loader, num_epochs, criterion, optimizer):
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in train_loader:
            if batch is None or batch.x is None or batch.y is None:
                continue
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

        acc = correct / total * 100 if total > 0 else 0
        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Train Acc = {acc:.2f}%")

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None or batch.x is None or batch.y is None:
                    continue
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)

        val_acc = val_correct / val_total * 100 if val_total > 0 else 0
        print(f"           Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

def test(model, device, test_loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            if batch is None or batch.x is None or batch.y is None:
                continue
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    acc = correct / total * 100 if total > 0 else 0
    print(f"Test Loss: {total_loss:.4f}, Test Acc: {acc:.2f}%")
    return acc

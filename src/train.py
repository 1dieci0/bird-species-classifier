import torch

def train_one_epoch(model, device, train_loader, optimizer, scheduler, criterion):
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)

        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    scheduler.step()

    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / train_total

    return (train_loss, train_acc)


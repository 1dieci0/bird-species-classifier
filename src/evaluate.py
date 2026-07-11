import torch

def validate_test(model, device, loader, criterion):
    model.eval()

    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            criterion_loss = criterion(outputs, labels)

            loss += criterion_loss.item()

            _, predicted = torch.max(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    loss /= len(loader)
    acc = 100 * correct / total

    return loss, acc


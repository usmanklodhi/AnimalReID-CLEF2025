import torch
import torch.nn.functional as F

def evaluate_soft_ensemble(model1, model2, dataloader, criterion, device, alpha=0.5):
    model1.eval()
    model2.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits1 = model1(images)
            logits2 = model2(images)
            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)
            blended_probs = alpha * probs1 + (1 - alpha) * probs2
            loss = criterion(blended_probs.log(), labels)
            total_loss += loss.item()
            preds = blended_probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy 
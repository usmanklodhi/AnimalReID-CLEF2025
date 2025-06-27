import torch
import csv
import os
import matplotlib.pyplot as plt

def train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, device, num_epochs, label_encoder, metrics_csv="epoch_metrics.csv", early_stopping=None):
    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_path = "best_model.pth"
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "learning_rate"])
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = validate(model, test_loader, criterion, device, epoch)
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        with open(metrics_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, train_loss, val_loss, val_acc, current_lr])
        if val_loss < best_val_loss and val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'label_encoder': label_encoder}, best_model_path)
        if early_stopping and early_stopping.should_stop(val_loss):
            break
    # Plotting
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("loss_plot.png")
    plt.close()
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.title("Validation Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig("accuracy_plot.png")
    plt.close()
    plt.plot(learning_rates, label="Learning Rate")
    plt.legend()
    plt.title("Learning Rate Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid()
    plt.savefig("learning_rate_plot.png")
    plt.close()

def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return running_loss / len(dataloader), accuracy 
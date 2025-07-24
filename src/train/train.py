import torch
import csv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, device, num_epochs, label_encoder, output_dir, metrics_csv="epoch_metrics.csv", early_stopping=None):
    os.makedirs(output_dir, exist_ok=True)
    metrics_csv = os.path.join(output_dir, os.path.basename(metrics_csv))
    best_model_path = os.path.join(output_dir, "best_model.pth")
    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    acc_plot_path = os.path.join(output_dir, "accuracy_plot.png")
    lr_plot_path = os.path.join(output_dir, "learning_rate_plot.png")

    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "learning_rate"])
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, labels in batch_iter:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            batch_iter.set_postfix(loss=loss.item())
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
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
        if val_loss < best_val_loss and val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'label_encoder': label_encoder}, best_model_path)
        if early_stopping and early_stopping.should_stop(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
    # Plotting
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(loss_plot_path)
    plt.close()
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.title("Validation Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(acc_plot_path)
    plt.close()
    plt.plot(learning_rates, label="Learning Rate")
    plt.legend()
    plt.title("Learning Rate Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid()
    plt.savefig(lr_plot_path)
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
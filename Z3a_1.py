import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import numpy as np

# Nastavenie zariadenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformácie pre MNIST: premena na Tensor a normalizácia
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Načítanie trénovacích a testovacích dát
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Vytvorenie DataLoader-ov
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definícia modelu - Viacvrstvový perceptrón
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Funkcia na vyhodnotenie modelu
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    avg_loss = test_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return avg_loss, accuracy, all_preds, all_labels

# Funkcia na trénovanie modelu s daným optimalizátorom
def train_model(optimizer_name='sgd', epochs=10):
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_name == 'sgd_momentum':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError("Neznámy optimizer: {}".format(optimizer_name))
    
    print(f"\nTrénujeme model s {optimizer_name.upper()}")

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        test_loss, test_acc, _, _ = evaluate(model, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f"Epoche {epoch}/{epochs}, Trénovacia strata: {train_loss:.4f}, Testovacia strata: {test_loss:.4f}, Presnosť: {test_acc:.2f}%")
        
    return model, train_losses, test_losses, test_accuracies

# Tréning troch modelov
epochs = 10
model_sgd, train_losses_sgd, test_losses_sgd, test_accuracies_sgd = train_model('sgd', epochs=epochs)
model_sgd_m, train_losses_sgd_m, test_losses_sgd_m, test_accuracies_sgd_m = train_model('sgd_momentum', epochs=epochs)
model_adam, train_losses_adam, test_losses_adam, test_accuracies_adam = train_model('adam', epochs=epochs)

# Porovnanie presností na konci tréningu
final_acc_sgd = test_accuracies_sgd[-1]
final_acc_sgd_m = test_accuracies_sgd_m[-1]
final_acc_adam = test_accuracies_adam[-1]

best_model = None
best_opt = None
best_acc = 0.0

if final_acc_sgd > best_acc:
    best_acc = final_acc_sgd
    best_model = model_sgd
    best_opt = "SGD"

if final_acc_sgd_m > best_acc:
    best_acc = final_acc_sgd_m
    best_model = model_sgd_m
    best_opt = "SGD_momentum"

if final_acc_adam > best_acc:
    best_acc = final_acc_adam
    best_model = model_adam
    best_opt = "Adam"

print(f"\nNajlepší model je trénovaný s {best_opt} s presnosťou {best_acc:.2f}% na testovacej množine.")

# Grafy z priebehu tréningu (Train Loss, Test Loss, Test Accuracy)
def plot_metrics(train_losses, test_losses, test_accuracies, title):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_metrics(train_losses_sgd, test_losses_sgd, test_accuracies_sgd, 'SGD')
plot_metrics(train_losses_sgd_m, test_losses_sgd_m, test_accuracies_sgd_m, 'SGD Momentum')
plot_metrics(train_losses_adam, test_losses_adam, test_accuracies_adam, 'Adam')

# Confusion matrix pre najlepší model
_, _, preds, labels = evaluate(best_model, test_loader)
cm = confusion_matrix(labels, preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_opt}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

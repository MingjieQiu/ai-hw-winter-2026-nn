import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============ DATA LOADING ============
def get_data_loaders(augment=False):
    """Load MNIST dataset"""
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=False)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


# MODELS
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.patch_size = 4
        self.num_patches = (28 // self.patch_size) ** 2
        self.patch_dim = self.patch_size * self.patch_size
        self.embed_dim = 64

        self.patch_embed = nn.Linear(self.patch_dim, self.embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4,
                                                   dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(self.embed_dim, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, 1, self.num_patches, -1).squeeze(1)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


# Test
def test_model(model, test_loader):
    """Test model and return accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Train
def train_model(model, train_loader, test_loader, model_name, epochs=10):
    """Train a model and return results"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch + 1}/{epochs}')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Testing
        accuracy = test_model(model, test_loader)
        test_accuracies.append(accuracy)

        print(f'{model_name} - Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Test Accuracy = {accuracy:.2f}%')

    return train_losses, test_accuracies


if __name__ == '__main__':
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(augment=False)

    # Define models to train
    models_to_train = {
        'MLP': MLP(),
        'CNN': CNN(),
        'Transformer': TransformerEncoder()
    }

    results = {}

    # Train all models
    for model_name, model in models_to_train.items():
        print(f"\n{'=' * 10} Training {model_name} {'=' * 10}")
        model = model.to(device)
        losses, accs = train_model(model, train_loader, test_loader, model_name, epochs=10)
        results[model_name] = {'final_accuracy': accs[-1], 'accuracies': accs}
        torch.save(model.state_dict(), f'models/{model_name.lower()}.pth')

    # Save results
    with open('results/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Print summary
    print("\n" + "=" * 10 + " FINAL RESULTS " + "=" * 10)
    for model_name, res in results.items():
        print(f"{model_name}: {res['final_accuracy']:.2f}%")

    # Plot results
    plt.figure(figsize=(10, 5))
    for model_name, res in results.items():
        plt.plot(res['accuracies'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Comparison on MNIST')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/accuracy_comparison.png')
    print("\nPlot saved to results/accuracy_comparison.png")

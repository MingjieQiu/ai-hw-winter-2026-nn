import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Data Loading
def get_data_loaders(augment=False):
    """
    Load and prepare MNIST dataset with data loaders. (60,000 training images, 10,000 test images)
    Images are 28x28 grayscale of handwritten digits (0-9)
    Args:
        augment: If True, applies data augmentation to training data (random rotation and translation)
    Returns:
        tuple: (train_loader, test_loader) for batched data iteration
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # Rotate images ±10 degrees
            transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Shift images by 10%
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

    train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


# MODELS
class MLP(nn.Module):
    """
    Simple feedforward neural network, with fully connected layers.
    Input (784) → FC(128) → ReLU → Dropout → FC(64) → ReLU → Dropout → FC(10)
    """

    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() # Flatten 28x28 image into 784-dimensional vector
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer: 784 → 128 neurons
        self.fc2 = nn.Linear(128, 64)  # Hidden layer: 128 → 64 neurons
        self.fc3 = nn.Linear(64, 10)  # Output layer: 64 → 10 classes (digits 0-9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """Forward pass"""
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    """
    Convolutional neural network designed for image data.
    Conv(32) → ReLU → MaxPool → Conv(64) → ReLU → MaxPool → FC(128) → FC(10)
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1→32 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32→64 feature maps
        self.pool = nn.MaxPool2d(2, 2)  # Halves width and height
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flatten convolutional output
        self.fc2 = nn.Linear(128, 10)  # Final classification layer

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """Forward pass"""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoder(nn.Module):
    """
    Vision Transformer encoder using self-attention mechanism.
    Patch Embedding → Positional Encoding → Transformer Layers → Classification
    """

    def __init__(self):
        super(TransformerEncoder, self).__init__()
        # Divide 28x28 image into 4x4 patches → 49 patches total
        self.patch_size = 4
        self.num_patches = (28 // self.patch_size) ** 2  # (28/4)^2 = 49 patches
        self.patch_dim = self.patch_size * self.patch_size  # 4*4 = 16 pixels per patch
        self.embed_dim = 64  # Embedding dimension for each patch
        self.patch_embed = nn.Linear(self.patch_dim, self.embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,  # Embedding dimension
            nhead=4,  # Number of attention heads
            dim_feedforward=256,  # Hidden dimension in feedforward network
            dropout=0.1,  # Dropout rate
            batch_first=True  # Batch dimension comes first
        )

        # Stack 2 transformer encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Final classification layer
        self.fc = nn.Linear(self.embed_dim, 10)

    def forward(self, x):
        """Forward pass: patchify → embed → attend → classify"""
        batch_size = x.size(0)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, 1, self.num_patches, -1).squeeze(1)
        x = self.patch_embed(x)  # [batch, 49, 64]
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)  # [batch, 64] → [batch, 10]
        return x


# Testing
def test_model(model, test_loader):
    """
    Evaluate model accuracy on test dataset.
    Args:
        model: Model to test
        test_loader: DataLoader containing test data
    Returns:
        float: Test accuracy percentage (0-100)
    """
    # Set model to evaluation mode
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # Disables gradient calculation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Training
def train_model(model, train_loader, test_loader, model_name, epochs=10):
    """
    Train the neural network model.
    Args:
        model: Neural network to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        model_name: Name of the model
        epochs: Number of training epochs
    Returns:
        tuple: (train_losses, test_accuracies): lists of metrics per epoch
    """
    criterion = nn.CrossEntropyLoss()  # rossEntropyLoss combines softmax + negativ

    # Automatically adjusts learning rate for each parameter
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
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(augment=False)

    models_to_train = {
        'MLP': MLP(),
        'CNN': CNN(),
        'Transformer': TransformerEncoder()
    }

    results = {}

    for model_name, model in models_to_train.items():
        print(f"\n{'=' * 10} Training {model_name} {'=' * 10}")
        model = model.to(device)
        losses, accuracy = train_model(model, train_loader, test_loader, model_name, epochs=10)

        results[model_name] = {
            'final_accuracy': accuracy[-1],
            'epochs_accuracies': accuracy
        }
        torch.save(model.state_dict(), f'models/{model_name.lower()}.pth')

    with open('results/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 10 + " FINAL RESULTS " + "=" * 10)
    for model_name, res in results.items():
        print(f"{model_name}: {res['final_accuracy']:.2f}%")

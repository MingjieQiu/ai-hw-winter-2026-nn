import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import MLP, CNN, TransformerEncoder
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_test_data():
    """Load MNIST test dataset"""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return test_loader


def test_model(model, test_loader, model_name):
    """Test a saved model"""
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
    print(f"{model_name} Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':
    test_loader = load_test_data()
    results = {}

    # Test MLP
    mlp = MLP().to(device)
    mlp.load_state_dict(torch.load('models/mlp.pth'))
    results['MLP'] = test_model(mlp, test_loader, "MLP")

    # Test CNN
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load('models/cnn.pth'))
    results['CNN'] = test_model(cnn, test_loader, "CNN")

    # Test Transformer
    transformer = TransformerEncoder().to(device)
    transformer.load_state_dict(torch.load('models/transformer.pth'))
    results['Transformer'] = test_model(transformer, test_loader, "Transformer")

    # Save test results
    with open('results/test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nTest results saved to results/test_results.json")
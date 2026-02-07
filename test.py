import torch
from train import MLP, CNN, TransformerEncoder, get_data_loaders, test_model, device
import json
import os

if __name__ == '__main__':
    """Loads pre-trained models and evaluates on the test set."""
    _, test_loader = get_data_loaders(augment=False)

    models_to_test = {
        'MLP': MLP(),
        'CNN': CNN(),
        'Transformer': TransformerEncoder()
    }

    results = {}

    for model_name, model in models_to_test.items():
        model = model.to(device)

        # Load pre-trained weights from saved './models/'
        model.load_state_dict(torch.load(f'models/{model_name.lower()}.pth'))

        accuracy = test_model(model, test_loader)
        results[model_name] = accuracy
        print(f"{model_name} Test Accuracy: {accuracy:.2f}%")

    os.makedirs('results', exist_ok=True)
    with open('results/test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nTest results saved to results/test_results.json")
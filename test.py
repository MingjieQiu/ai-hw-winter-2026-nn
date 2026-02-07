import torch
from train import MLP, CNN, TransformerEncoder, get_data_loaders, test_model
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Load test data
    _, test_loader = get_data_loaders(augment=False)

    # Define models
    models_to_test = {
        'MLP': MLP(),
        'CNN': CNN(),
        'Transformer': TransformerEncoder()
    }

    results = {}

    # Test all models
    for model_name, model in models_to_test.items():
        model = model.to(device)
        model.load_state_dict(torch.load(f'models/{model_name.lower()}.pth'))
        accuracy = test_model(model, test_loader)
        results[model_name] = accuracy
        print(f"{model_name} Test Accuracy: {accuracy:.2f}%")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nTest results saved to results/test_results.json")
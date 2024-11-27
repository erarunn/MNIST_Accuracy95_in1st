import torch
from Train import Net, train, test
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
from utils.augmentation import MNISTAugmentation

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_predictions_shape(model, device):
    # Test if model outputs correct shape
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    output = model(dummy_input)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    return True

def test_model_gradient_flow(model, device):
    # Test if gradients are flowing properly
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    dummy_target = torch.tensor([5]).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    
    output = model(dummy_input)
    loss = torch.nn.functional.nll_loss(output, dummy_target)
    loss.backward()
    
    # Check if gradients exist and are not zero
    has_gradients = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            if param.grad.sum() != 0:
                has_gradients = True
                break
    
    assert has_gradients, "No gradients are flowing in the model"
    return True

def test_model_batch_processing(model, device):
    # Test if model can handle different batch sizes
    batch_sizes = [1, 32, 64, 128]
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)
        output = model(dummy_input)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"
    return True

def main():
    # Setup device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Create model
    model = Net().to(device)
    
    # Run all tests
    print("Running tests...")
    
    # 1. Parameter count test
    param_count = count_parameters(model)
    print(f"Parameter count: {param_count}")
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"
    
    # 2. New tests
    print("Testing model prediction shape...")
    assert test_model_predictions_shape(model, device)
    
    print("Testing gradient flow...")
    assert test_model_gradient_flow(model, device)
    
    print("Testing batch processing...")
    assert test_model_batch_processing(model, device)
    
    # 3. Accuracy test
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # Setup augmentation
    augmentation = MNISTAugmentation()
    
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                 transform=augmentation.aug_transforms)
    
    # Visualize augmentations
    augmentation.visualize_augmentations(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=128, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=128, shuffle=True, **kwargs)
    
    # Train for 1 epoch
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(model, device, train_loader, optimizer, 1)
    
    # Test accuracy
    accuracy = test(model, device, test_loader)
    assert accuracy > 95.0, f"Model accuracy is {accuracy}%, should be > 95%"
    
    print("All tests passed!")

if __name__ == "__main__":
    main() 
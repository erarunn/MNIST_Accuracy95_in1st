import torch
from Train import Net, train, test
from torchvision import datasets, transforms
import torch.optim as optim

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Setup device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Create model
    model = Net().to(device)
    
    # Check parameter count
    param_count = count_parameters(model)
    print(f"Parameter count: {param_count}")
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"
    
    # Setup data
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
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
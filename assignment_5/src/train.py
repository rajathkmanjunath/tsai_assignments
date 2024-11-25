import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os
from tqdm import tqdm

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    
    # Training
    model.train()
    print(f"Training on {device}...")
    
    pbar = tqdm(train_loader, desc='Training')
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        running_loss = 0.9 * running_loss + 0.1 * loss.item()
        pbar.set_postfix({'loss': f'{running_loss:.4f}'})
    
    # Save model with timestamp
    os.makedirs('models', exist_ok=True)  # Create models directory if it doesn't exist
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')    
    # Also save as latest for easy reference
    latest_path = os.path.join('models', 'model_mnist_latest.pth')
    torch.save(model.state_dict(), latest_path)
    print(f"Also saved as {latest_path}")
    return latest_path

if __name__ == '__main__':
    train_model() 
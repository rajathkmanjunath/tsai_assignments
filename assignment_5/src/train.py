import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from model import MNISTModel
from datetime import datetime
import os
from tqdm import tqdm

class RotatedMNIST(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.rotation_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        # Convert to PIL image for rotation
        image_np = image.squeeze().numpy()
        # Create rotated version
        rotated_image = self.rotation_transform(image_np)
        return rotated_image, label

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading with augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading MNIST dataset...")
    original_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    rotated_dataset = RotatedMNIST(original_dataset)
    
    # Combine original and rotated datasets
    combined_dataset = ConcatDataset([original_dataset, rotated_dataset])
    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    # Rest of the training code remains the same
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    
    # Training
    model.train()
    print(f"Training on {device} with augmented dataset...")
    
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
    
    # Save model
    os.makedirs('models', exist_ok=True)
    latest_path = os.path.join('models', 'model_mnist_latest.pth')
    torch.save(model.state_dict(), latest_path)
    print(f"Model saved as {latest_path}")
    return latest_path

if __name__ == '__main__':
    train_model() 
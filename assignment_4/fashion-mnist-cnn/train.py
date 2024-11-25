import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import time
import random
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Input: [batch_size, 1, 28, 28]
        out = self.layer1(x)     # [batch_size, 32, 14, 14]
        out = self.layer2(out)   # [batch_size, 64, 7, 7]
        out = self.layer3(out)   # [batch_size, 128, 3, 3]
        out = self.layer4(out)   # [batch_size, 256, 1, 1]
        out = out.view(out.size(0), -1)  # [batch_size, 256]
        out = self.dropout(torch.relu(self.fc1(out)))
        out = self.fc2(out)      # [batch_size, 10]
        return out

def save_training_state(epoch, loss, accuracy):
    with open('static/training_state.json', 'w') as f:
        json.dump({
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy
        }, f)

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    batch_size = 64  # Reduced batch size for better stability
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('data', train=False, transform=transform)

    # Add drop_last=True to ensure all batches have the same size
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            drop_last=True)  # This drops incomplete batches
    test_loader = DataLoader(test_dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           drop_last=True)

    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 10
    print_every = 50  # Print stats every 50 batches
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            # Ensure both images and labels are on the same device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % print_every == print_every - 1:
                accuracy = 100 * correct / total
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/print_every:.4f}, '
                      f'Accuracy: {accuracy:.2f}%')
                
                save_training_state(epoch + 1, running_loss/print_every, accuracy)
                running_loss = 0.0
                correct = 0
                total = 0

    # Save final model
    torch.save(model.state_dict(), 'model.pth')

    # Generate and save test results
    model.eval()
    test_images = []
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    with torch.no_grad():
        for images, labels in test_loader:
            test_images.extend(list(zip(images, labels)))
            if len(test_images) >= 10:
                break

    random_samples = random.sample(test_images, 10)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for idx, (image, label) in enumerate(random_samples):
        image = image.to(device)
        output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        
        axes[idx].imshow(image.cpu().squeeze(), cmap='gray')
        axes[idx].set_title(f'Pred: {class_names[pred]}\nTrue: {class_names[label]}')
        axes[idx].axis('off')

    plt.tight_layout()
    
    # Save plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode()

    with open('static/results.txt', 'w') as f:
        f.write(graphic)

if __name__ == '__main__':
    main() 
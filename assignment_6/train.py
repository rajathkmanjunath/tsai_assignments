import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_model():
    logger = setup_logging()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    logger.info("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    logger.info(f"Dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {total_params} parameters")
    
    best_accuracy = 0.0
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Evaluate after each epoch
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)
        
        test_accuracy = 100. * test_correct / test_total
        avg_test_loss = test_loss / test_total
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Training Loss: {running_loss/(len(train_loader)):.4f}, '
                   f'Training Accuracy: {100.*correct/total:.2f}%')
        logger.info(f'Test Loss: {avg_test_loss:.4f}, '
                   f'Test Accuracy: {test_accuracy:.2f}%')
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f'New best model saved with accuracy: {best_accuracy:.2f}%')
    
    # Final evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    final_correct = 0
    final_total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Final Evaluation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            final_correct += pred.eq(target.view_as(pred)).sum().item()
            final_total += target.size(0)
    
    final_accuracy = 100. * final_correct / final_total
    logger.info(f'Final Test Accuracy: {final_accuracy:.2f}%')
    logger.info(f'Total parameters: {total_params}')
    
    return final_accuracy, total_params

if __name__ == "__main__":
    accuracy, params = train_model()
    # Ensure consistent output format for GitHub Actions
    print("-" * 50)
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"Total parameters: {params}")
    print("-" * 50) 
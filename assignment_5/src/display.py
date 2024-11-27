import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def display_mnist_samples(num_images=5, rows=1):
    """
    Display sample images from MNIST dataset
    Args:
        num_images: Number of images to display
        rows: Number of rows in the plot grid
    """
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Calculate number of columns needed
    cols = (num_images + rows - 1) // rows
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot images
    for i in range(num_images):
        row = i // cols
        col = i % cols
        
        # Get an image and its label
        image, label = dataset[i]
        
        # Convert tensor to numpy and reshape
        image = image.squeeze().numpy()
        
        # Display image
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Label: {label}')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Display 10 images in 2 rows
    display_mnist_samples(10, 2)
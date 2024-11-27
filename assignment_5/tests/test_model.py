import unittest
import torch
from src.model import MNISTModel
from src.utils import count_parameters, evaluate_model
from torchvision import datasets, transforms
import torch.nn.functional as F

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTModel()
        
    def test_parameter_count(self):
        param_count = count_parameters(self.model)
        print(f"Model has {param_count} parameters")
        self.assertLess(param_count, 25000, f"Model has {param_count} parameters, should be < 100000")
        
    def test_input_shape(self):
        test_input = torch.randn(1, 1, 28, 28)
        try:
            output = self.model(test_input)
            self.assertEqual(output.shape[1], 10, "Output should have 10 classes")
        except Exception as e:
            self.fail(f"Model failed to process 28x28 input: {str(e)}")
            
    def test_accuracy(self):
        self.model.load_state_dict(torch.load('models/model_mnist_latest.pth'))
        accuracy = evaluate_model(self.model)
        print(f"Model accuracy: {accuracy:.2f}")
        self.assertGreater(accuracy, 0.95, f"Model accuracy {accuracy:.2f} should be > 0.95")
        
    def test_model_predictions_shape(self):
        """Test if model outputs correct prediction shape"""
        batch_size = 4
        test_input = torch.randn(batch_size, 1, 28, 28)  # MNIST image shape
        output = self.model(test_input)
        
        # Should output batch_size x num_classes (10 for MNIST)
        self.assertEqual(output.shape, (batch_size, 10))
        
    def test_model_rotation_invariance(self):
        """Test if model maintains reasonable accuracy with rotated inputs"""
        # Load a sample image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        image, true_label = dataset[0]
        
        # Get prediction for original image
        original_pred = self.model(image.unsqueeze(0))
        original_class = original_pred.argmax(dim=1).item()
        
        # Rotate image by 15 degrees
        rotation_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(10, 10)),  # Fixed 10 degree rotation
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        rotated_image = rotation_transform(image.squeeze().numpy())
        rotated_pred = self.model(rotated_image.unsqueeze(0))
        rotated_class = rotated_pred.argmax(dim=1).item()
        
        # Predictions should match for small rotations
        self.assertEqual(original_class, rotated_class)
        
    def test_contrast_invariance(self):
        """Test if model maintains consistent predictions with contrast changes"""
        # Load a sample image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        image, true_label = dataset[0]
        
        # Get prediction for original image
        with torch.no_grad():
            original_pred = self.model(image.unsqueeze(0))
            original_class = original_pred.argmax(dim=1).item()
        
        # Increase contrast
        contrast_factor = 1.5
        contrast_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(contrast=contrast_factor),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Apply contrast change
        high_contrast_image = contrast_transform(image.squeeze().numpy())
        
        # Get prediction for high contrast image
        with torch.no_grad():
            contrast_pred = self.model(high_contrast_image.unsqueeze(0))
            contrast_class = contrast_pred.argmax(dim=1).item()
        
        # Compare predictions
        self.assertEqual(original_class, contrast_class, 
                        f"Model predictions changed after contrast adjustment: "
                        f"Original: {original_class}, Contrast-adjusted: {contrast_class}")
        
        # Optional: Check confidence difference
        original_confidence = torch.nn.functional.softmax(original_pred, dim=1).max().item()
        contrast_confidence = torch.nn.functional.softmax(contrast_pred, dim=1).max().item()
        confidence_diff = abs(original_confidence - contrast_confidence)
        
        # Confidence shouldn't change drastically
        self.assertLess(confidence_diff, 0.3, 
                        f"Confidence changed too much after contrast adjustment: "
                        f"Difference of {confidence_diff:.3f}")

if __name__ == '__main__':
    unittest.main() 
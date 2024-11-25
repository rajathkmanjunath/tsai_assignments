import unittest
import torch
from src.model import MNISTModel
from src.utils import count_parameters, evaluate_model

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTModel()
        
    def test_parameter_count(self):
        param_count = count_parameters(self.model)
        self.assertLess(param_count, 100000, f"Model has {param_count} parameters, should be < 100000")
        
    def test_input_shape(self):
        test_input = torch.randn(1, 1, 28, 28)
        try:
            output = self.model(test_input)
            self.assertEqual(output.shape[1], 10, "Output should have 10 classes")
        except Exception as e:
            self.fail(f"Model failed to process 28x28 input: {str(e)}")
            
    def test_accuracy(self):
        self.model.load_state_dict(torch.load('model_mnist_latest.pth'))
        accuracy = evaluate_model(self.model)
        self.assertGreater(accuracy, 0.8, f"Model accuracy {accuracy:.2f} should be > 0.8")

if __name__ == '__main__':
    unittest.main() 
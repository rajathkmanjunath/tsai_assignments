import unittest
import torch
from src.model import MNISTModel
from src.utils import count_parameters, evaluate_model

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

if __name__ == '__main__':
    unittest.main() 
from .base_processor import BaseProcessor
import cv2
import numpy as np
import albumentations as A
import random
from pathlib import Path

class ImageProcessor(BaseProcessor):
    def __init__(self):
        # Preprocessing pipeline
        self.preprocessor = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        self.augmentation_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(
                var_limit=(5.0, 15.0),
                p=0.5
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.5
            ),
            A.Blur(
                blur_limit=3,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
                p=0.5
            )
        ]

    def preprocess(self, image_path):
        """
        Preprocess the image:
        - Resize to standard size
        - Normalize
        """
        # Read image in BGR
        image = cv2.imread(str(image_path))
        
        # Convert to RGB for processing and web display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        preprocessed = self.preprocessor(image=image)['image']
        
        # Convert back to uint8 for saving
        preprocessed = (preprocessed * 255).astype(np.uint8)
        
        return preprocessed  # Keep in RGB format

    def augment(self, image):
        """
        Apply random augmentations
        """
        # Image is already in RGB format
        
        # Randomly select number of augmentations
        num_augmentations = random.randint(1, 2)
        selected_transforms = random.sample(self.augmentation_transforms, num_augmentations)
        
        # Create augmentation pipeline with selected transforms
        augmentor = A.Compose(selected_transforms)
        
        # Apply augmentations
        augmented = augmentor(image=image)['image']
        
        # Get list of applied transformations
        applied_techniques = []
        for transform in selected_transforms:
            name = type(transform).__name__
            if name == 'HorizontalFlip':
                applied_techniques.append("Horizontal Flip")
            elif name == 'RandomBrightnessContrast':
                applied_techniques.append("Brightness/Contrast Adjustment")
            elif name == 'RandomRotate90':
                applied_techniques.append("90° Rotation")
            elif name == 'GaussNoise':
                applied_techniques.append("Subtle Noise")
            elif name == 'RandomGamma':
                applied_techniques.append("Gamma Adjustment")
            elif name == 'Blur':
                applied_techniques.append("Slight Blur")
            elif name == 'ShiftScaleRotate':
                applied_techniques.append("Shift/Scale/Rotate")
        
        return augmented, applied_techniques  # Keep in RGB format

    def save_image(self, image, save_path):
        """Save the image in RGB format for web display"""
        # PIL/imageio expects RGB format for web-compatible images
        import imageio
        imageio.imwrite(str(save_path), image)
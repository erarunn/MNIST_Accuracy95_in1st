import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

class MNISTAugmentation:
    def __init__(self):
        # Base transforms for converting to tensor and normalization
        self.base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Augmentation transforms
        self.aug_transforms = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Create directory for saving augmented samples
        os.makedirs('augmentation_samples', exist_ok=True)
    
    def visualize_augmentations(self, dataset, num_samples=5):
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2*num_samples))
        
        for i in range(num_samples):
            # Get original image (before any transforms)
            img, _ = dataset.data[i], dataset.targets[i]
            
            # Convert to PIL Image for transforms
            img_pil = Image.fromarray(img.numpy())
            
            # Original image (with base transforms)
            img_tensor = self.base_transforms(img_pil)
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Augmented image
            img_aug = self.aug_transforms(img_pil)
            axes[i, 1].imshow(img_aug.squeeze(), cmap='gray')
            axes[i, 1].set_title('Augmented')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('augmentation_samples/augmentation_examples.png')
        plt.close() 
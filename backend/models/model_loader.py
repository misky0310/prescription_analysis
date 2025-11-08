"""
Model loading and transformation utilities
"""
import torch
import numpy as np
import os
from torchvision import transforms
from models.medicine_classifier import MedicineClassifier
import config.settings as settings


def load_model(model_path, label_encoder_path, device):
    """Load trained model and label encoder"""
    print('='*70)
    print('🔄 Loading Model')
    print('='*70)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
    
    label_classes = np.load(label_encoder_path, allow_pickle=True)
    num_classes = len(label_classes)
    
    # Update global label_classes
    settings.label_classes = label_classes
    
    print(f'📊 Number of classes: {num_classes}')
    print(f'🏷️  Medicine classes: {", ".join(label_classes)}')
    
    model = MedicineClassifier(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f'✅ Model loaded successfully!')
    print(f'🎯 Trained accuracy: {checkpoint["accuracy"]:.2f}%')
    print(f'📅 Trained at epoch: {checkpoint["epoch"] + 1}')
    
    return model, label_classes


def get_transform():
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
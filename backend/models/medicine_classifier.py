"""
Medicine Classifier Model Architecture
"""
import torch.nn as nn
from torchvision import models


class MedicineClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MedicineClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
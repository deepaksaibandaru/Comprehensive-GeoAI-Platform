"""Standalone chip-level Classic CNN classifier for Breain.ai."""
import torch
import torch.nn as nn

class ClassicCNN(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, base=32, dropout=0.0, **kwargs):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(n_channels,base,3,padding=1),nn.BatchNorm2d(base),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(base,base*2,3,padding=1),nn.BatchNorm2d(base*2),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(base*2,base*4,3,padding=1),nn.BatchNorm2d(base*4),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(base*4,base*8,3,padding=1),nn.BatchNorm2d(base*8),nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1))
        self.dropout=nn.Dropout(dropout) if dropout else nn.Identity()
        self.classifier=nn.Linear(base*8,max(2,int(n_classes)))
        self.geoai_model_type='classic_cnn'; self.geoai_architecture='ClassicCNNClassifier'
    def forward(self,x): return self.classifier(self.dropout(self.features(x).flatten(1)))

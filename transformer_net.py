import torch
import torch.nn as nn

class MaskNoMaskClassifier(nn.Module):
  def __init__(self):
    super(MaskNoMaskClassifier, self).__init__()

    self.sequential = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3), # (32, 148, 148)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # (32, 74, 74),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # (64, 72, 72)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # (64, 36, 36)
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), # (128, 34, 34)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # (128, 17, 17)
        nn.Flatten(), # 36992
    )

    self.classifier = nn.Sequential(
        nn.Linear(in_features=36992, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=1)
    )

  def forward(self, features):
    features = self.sequential(features)
    features = self.classifier(features)

    return features
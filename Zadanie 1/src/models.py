import torch.nn as nn
from torchvision.models import resnet18

class FaceID_CNN(nn.Module):
    def __init__(self):
        super(FaceID_CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class Ready_faceID_CNN(nn.Module):
    def __init__(self):
        super(Ready_faceID_CNN, self).__init__()
        model = resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )
    
        self.model = model

    def forward(self, x):
        return self.model(x)

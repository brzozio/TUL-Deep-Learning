from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import os

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH : str  = os.path.dirname(SRC_PATH) + '\\csv'
JSON_PATH : str = os.path.dirname(SRC_PATH) + '\\json'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

celeba_train = CelebA(root=CSV_PATH, split='train', download=False, transform=transform)
celeba_val   = CelebA(root=CSV_PATH, split='valid', download=False, transform=transform)
celeba_test  = CelebA(root=CSV_PATH, split='test', download=False, transform=transform)

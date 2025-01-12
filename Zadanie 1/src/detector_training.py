import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SRC_PATH), "csv")
MODEL_PATH = os.path.join(os.path.dirname(SRC_PATH), "models")
IMAGES_PATH = os.path.join(DATA_PATH, "widerface", "WIDER_train", "WIDER_train", "images")
ANNOTATIONS_FILE = os.path.join(DATA_PATH, "widerface", "wider_face_split", "wider_face_train_bbx_gt.txt")

class WiderFaceDataset(Dataset):
    def __init__(self, images_folder, annotations_file, transform=None):
        self.images_folder = images_folder
        self.annotations_file = annotations_file
        self.transform = transform
        self.image_data = self.parse_annotations()

    def parse_annotations(self):
        data = []
        with open(self.annotations_file, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        idx = 0
        while idx < len(lines):
            try:
                img_path = lines[idx]
                img_full_path = os.path.join(self.images_folder, img_path)

                if not img_path.endswith(".jpg"):
                    idx += 1
                    continue

                idx += 1
                num_faces = int(lines[idx])

                boxes = []
                for _ in range(num_faces):
                    idx += 1
                    bbox_data = lines[idx].split()
                    if len(bbox_data) < 4:
                        continue
                    x, y, w, h = map(int, bbox_data[:4])
                    boxes.append([x, y, x + w, y + h])

                data.append({"image_path": img_full_path, "boxes": boxes})
                idx += 1

            except (ValueError, IndexError):
                idx += 1  # Skip invalid lines
        return data

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        while True:  # Loop until a valid image is found
            img_info = self.image_data[idx]
            img_path = img_info["image_path"]
            boxes = torch.tensor(img_info["boxes"], dtype=torch.float32)

            # Filter invalid boxes if boxes is not empty
            if boxes.numel() > 0:
                boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]

            # If no valid boxes remain, skip this image
            if len(boxes) == 0:
                print(f"No valid boxes for image: {img_path}. Skipping...")
                idx = (idx + 1) % len(self)  # Move to the next index cyclically
                continue

            labels = torch.ones((len(boxes),), dtype=torch.int64)  # All faces labeled as '1'

            try:
                img = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                idx = (idx + 1) % len(self)  # Move to the next index cyclically
                continue

            # New image dimensions after resize
            original_width, original_height = img.size

            if self.transform:
                img = self.transform(img)

            # Check if img is still a PIL image after transformation
            if isinstance(img, Image.Image):
                new_width, new_height = img.size
            else:
                # In case transform returns a tensor, get the size from tensor shape
                new_width, new_height = img.shape[1], img.shape[2]

            scale_x = new_width / original_width
            scale_y = new_height / original_height

            # Resize bounding boxes to match the new image size
            boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y], dtype=torch.float32)

            target = {"boxes": boxes, "labels": labels}
            return img, target


def collate_fn(batch):
    return tuple(zip(*batch))

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss
    else:
        return model, optimizer, 0, 0

# Create a function to log losses into a text file
def log_losses(epoch, train_loss, val_loss, log_file):
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")

def train_model(model, dataset, device, epochs=30, batch_size=4, lr=0.0005, save_path=None, checkpoint_path=None, patience=3, logfile=None):

    train_dataset = dataset
    val_dataset   = dataset

    train_size = int(0.8 * len(dataset.image_data))
    val_size = len(dataset.image_data) - train_size
    # val_dataset.image_data, train_dataset.image_data = random_split(dataset.image_data, [train_size, val_size])
    # train_images, val_images = random_split(dataset.image_data, [train_size, val_size])
    # train_dataset.image_data = train_images
    # val_dataset.image_data = val_images
    # print(f"Train size: {len(train_dataset.image_data)}, val size: {len(val_dataset.image_data)}")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=6)
    val_data_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=6)
    
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        print(f"Starting Epoch : {epoch + 1}")
        model.train()
        total_loss = 0

        for images, targets in train_data_loader:
            # Skip empty batches
            if not images or not targets:
                print("Empty batch. Skipping...")
                continue

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Handle NaN or invalid losses
            if not torch.isfinite(losses):
                print("Invalid loss encountered. Skipping this batch...")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        

        # Compute average loss for the epoch
        avg_loss = total_loss / len(train_data_loader)
        elapsed_time = time.time() - start_time  # Measure time for the epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss train: {avg_loss:.4f}, Time: {elapsed_time:.2f} seconds")

        # Validation loop
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():  # No gradients are needed for validation
            for images, targets in val_data_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_val_loss += losses.item()

        avg_val_loss = total_val_loss / len(val_data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss validation: {avg_val_loss:.4f}")

        # Log train and validation loss to the text file
        log_losses(epoch, avg_loss, avg_val_loss, logfile)

        # Check for improvement in validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            if save_path:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, save_path)
                print(f"Checkpoint saved to {save_path}")
        else:
            epochs_without_improvement += 1

        # Early stopping if no improvement for `patience` epochs
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)


if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((300, 300)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = WiderFaceDataset(images_folder=IMAGES_PATH, annotations_file=ANNOTATIONS_FILE, transform=transform)
    print(f"Dataset len is: {len(dataset.image_data)}")

    # dataset.image_data = dataset.image_data[:20000]

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Working on device: {device}")

    checkpoint_path = os.path.join(MODEL_PATH, "faster_rcnn_widerface.pth")
    save_path = os.path.join(MODEL_PATH, "faster_rcnn_widerface.pth")

    train_model(model, dataset, device, epochs=10, save_path=save_path, checkpoint_path=checkpoint_path, logfile=os.path.join(MODEL_PATH, "face_detector_loss.txt"))

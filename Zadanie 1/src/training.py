import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchmetrics.classification import BinaryAccuracy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from models import FaceID_CNN, Ready_faceID_CNN
import os
from torchvision import datasets
from PIL import Image
import pandas as pd
from torchvision.datasets import ImageFolder
import torch
import cv2
from skimage.feature import Cascade
from torchvision import transforms
from PIL import Image
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

SRC_PATH:    str = os.path.dirname(os.path.abspath(__file__))
CSV_PATH:    str = os.path.dirname(SRC_PATH) + '\\csv'
JSON_PATH:   str = os.path.dirname(SRC_PATH) + '\\json'
MODELS_PATH: str = os.path.dirname(SRC_PATH) + '\\models'

TRAIN:          bool = True
CONTINUE_TRAIN: bool = False
MODEL_NAME:     str  = 'nasz' # "nasz" albo "torch"
CAMERA_TEST:    bool = True
CELEBA_TEST:    bool = False
WIDERFACE_TEST: bool = False

class FaceIDModel(pl.LightningModule):
    def __init__(self, model, train_loader, val_loader, lr=1e-4):
        super(FaceIDModel, self).__init__()
        self.model = model
        self.lr = lr
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

        self.train_loader = train_loader
        self.val_loader = val_loader

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.float().unsqueeze(1)
        loss = self.criterion(y_hat, y)
        acc = self.train_acc(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.float().unsqueeze(1)
        loss = self.criterion(y_hat, y)
        acc = self.val_acc(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    # def set_criterion(self, criterion):
        # self.criterion = criterion

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, labels, transform=None):
        self.img_folder = img_folder
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_folder, self.labels.iloc[idx]["image_id"])
        label = self.labels.iloc[idx][1:].values.astype("float32").item()
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

def prepare_data(traits, batch_size=32, img_size=128, num_workers=4):
        
    train_transform3 = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Assuming RGB
    ])

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Assuming RGB
    ])


    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_path     = os.path.join(CSV_PATH, "celeba")
    img_folder_path  = os.path.join(dataset_path, "img_align_celeba")
    attr_path        = os.path.join(dataset_path, "list_attr_celeba.txt")
    partition_path   = os.path.join(dataset_path, "list_eval_partition.txt")

    if not os.path.exists(img_folder_path) or not os.path.exists(attr_path) or not os.path.exists(partition_path):
        raise FileNotFoundError("The dataset folder or required files are missing.")

    attr_df = pd.read_csv(attr_path, sep=r'\s+', skiprows=1)
    attr_df = attr_df.reset_index().rename(columns={"index": "image_id"})
    attr_df["image_id"] = attr_df["image_id"].astype(str)

    attr_df         = attr_df[["image_id"] + traits]

    attr_df[traits] = (attr_df[traits] + 1) // 2

    # filtered_df     = attr_df[(attr_df[traits] > 0).any(axis=1)]

    partition_df = pd.read_csv(partition_path, sep=' ', header=None, names=["image_id", "partition"])
    partition_df["image_id"] = partition_df["image_id"].astype(str)

    # filtered_df = filtered_df.merge(partition_df, on="image_id")
    non_filtered_df = attr_df.merge(partition_df, on="image_id")

    train_df = non_filtered_df[non_filtered_df["partition"] == 0].drop(columns=["partition"])
    val_df   = non_filtered_df[non_filtered_df["partition"] == 1].drop(columns=["partition"])
    test_df  = non_filtered_df[non_filtered_df["partition"] == 2].drop(columns=["partition"])

    # print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One of the dataset splits is empty. Check your data files and partition file.")

    train_data   = CelebADataset(img_folder_path, train_df, train_transform)
    train_data_3 = CelebADataset(img_folder_path, train_df, train_transform3)
    combined_train_data = ConcatDataset([train_data, train_data_3])
     
    val_data   = CelebADataset(img_folder_path, val_df, val_test_transform)
    test_data  = CelebADataset(img_folder_path, test_df, val_test_transform)

    train_loader = DataLoader(combined_train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    pos_weight = torch.tensor([len(non_filtered_df) / (2 * non_filtered_df[traits].sum().sum())])

    # train_loader = iter(train_loader)

    return train_loader, val_loader, test_loader, pos_weight

def train(model) -> tuple:
    max_epochs = 100
    
    train_losses = []
    val_losses = []

    class LossLogger(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            train_loss = trainer.callback_metrics.get("train_loss")
            if train_loss:
                train_losses.append(train_loss.item())
        
        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss:
                val_losses.append(val_loss.item())

    model.train()
    loss_logger = LossLogger()

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODELS_PATH,
        filename=f"{MODEL_NAME}_{{epoch:03d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", 
        patience=3, 
        verbose=True, 
        mode="min"
    )

    faceid_trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stopping, checkpoint_callback, loss_logger],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None
    )

    faceid_trainer.fit(model)

    return (train_losses, val_losses)

def plot_losses(train_losses, val_losses):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    batch_size = 32
    img_size = 128
    if MODEL_NAME == "nasz":
        model_pre    = FaceID_CNN()
        # criterion    = nn.BCELoss()
        trait = ['Male']
        train_loader, val_loader, test_loader, pos_weight = prepare_data(batch_size=batch_size, img_size=img_size, traits=trait)
        model        = FaceIDModel(model_pre, train_loader, val_loader)
    elif MODEL_NAME == "torch":
        model_pre = Ready_faceID_CNN()
        # criterion    = nn.BCELoss()
        trait = ['Smiling']
        train_loader, val_loader, test_loader, pos_weight = prepare_data(batch_size=batch_size, img_size=img_size, traits=['Smiling'])
        model        = FaceIDModel(model_pre, train_loader, val_loader)

    if TRAIN:
        if CONTINUE_TRAIN:
            checkpoint_path = MODELS_PATH + f"\\{MODEL_NAME}.ckpt"

            checkpoint = torch.load(checkpoint_path)
            if "criterion.pos_weight" in checkpoint["state_dict"]:
                del checkpoint["state_dict"]["criterion.pos_weight"]
            torch.save(checkpoint, checkpoint_path)

            model = FaceIDModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model.model,
                train_loader=train_loader,
                val_loader=val_loader
            )
            criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            model.criterion = criterion
            train_losses, val_losses = train(model=model)

            plot_losses(train_losses, val_losses)
            
        else:
            criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            model.criterion = criterion
            train_losses, val_losses = train(model=model)

            plot_losses(train_losses, val_losses)

    else:
        if CELEBA_TEST == True:
            checkpoint_path = MODELS_PATH + f"\\{MODEL_NAME}.ckpt"

            checkpoint = torch.load(checkpoint_path)
            if "criterion.pos_weight" in checkpoint["state_dict"]:
                del checkpoint["state_dict"]["criterion.pos_weight"]
            torch.save(checkpoint, checkpoint_path)

            model = FaceIDModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model.model,
                train_loader=train_loader,
                val_loader=val_loader
            )
            criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            model.criterion = criterion

            model.eval()
            model.freeze()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    x, y = x.to(device), y.to(device)

                    outputs = model(x)
                    predictions = (outputs > 0.5).float()
                    all_predictions.append(predictions.cpu())
                    all_targets.append(y.cpu())

            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)

            # test_accuracy = (all_predictions == all_targets).float().mean().item()
            test_accuracy = 0
            
            for pred_i, pred in enumerate(all_predictions):
                if all_targets[pred_i] == pred:
                    test_accuracy += 1
            
            print(f"Test Accuracy: {100*(test_accuracy/len(all_predictions)):.4f}")

            cm = confusion_matrix(all_targets, all_predictions)

            # Plot the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix - {trait[0]}')
            plt.show()
        elif CAMERA_TEST == True:
            def detect(frame, detector):
                detections = detector.detect_multi_scale(
                    img=frame, scale_factor=1.2, step_ratio=1,
                    min_size=(128, 128), max_size=(200, 200)
                )
                boxes = []
                for detection in detections:
                    x = detection['c']
                    y = detection['r']
                    w = detection['width']
                    h = detection['height']
                    boxes.append((x, y, w, h))
                return boxes

            def draw(frame, boxes, padding=70):
                
                height, width, _ = frame.shape

                for x, y, w, h in boxes:
                    new_x = max(x - padding, 0)
                    new_y = max(y - padding, 0)
                    new_w = min(w + 2 * padding, width - new_x)
                    new_h = min(h + 2 * padding, height - new_y)

                    frame = cv2.rectangle(frame, 
                                        (new_x, new_y), 
                                        (new_x + new_w, new_y + new_h), 
                                        color=(255, 0, 0), 
                                        thickness=2)
                return frame

            def preprocess_face(image, img_size=128):
                """Preprocess the face image for model inference."""
                transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
                image = Image.fromarray(image)
                return transform(image).unsqueeze(0)
            
            face_cascade_file = CSV_PATH + "\\face.xml"
            detector = Cascade(face_cascade_file)

            checkpoint_path = f"{MODELS_PATH}\\{MODEL_NAME}.ckpt"
            checkpoint = torch.load(checkpoint_path)

            if "criterion.pos_weight" in checkpoint["state_dict"]:
                del checkpoint["state_dict"]["criterion.pos_weight"]

            torch.save(checkpoint, checkpoint_path)

            model = FaceIDModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model.model,
                train_loader=None,
                val_loader=None
            )

            criterion = nn.BCEWithLogitsLoss()
            model.criterion = criterion

            model.eval()
            model.freeze()

            cap = cv2.VideoCapture(0)
            skip = 5
            frame_count = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if frame_count % skip == 0:
                    boxes = detect(gray_frame, detector)

                frame = draw(frame, boxes)

                for x, y, w, h in boxes:
                    face = frame[y:y + h, x:x + w]
                    preprocessed_face = preprocess_face(face, img_size=img_size).to(device)

                    with torch.no_grad():
                        output = model(preprocessed_face)
                        label = f"{trait[0]}" if output > 0.5 else "Negative"

                    cv2.putText(
                        frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )

                cv2.imshow('FaceID Camera Test', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_count += 1

            cap.release()
            cv2.destroyAllWindows()

        elif WIDERFACE_TEST == True:
            def preprocess_widerface(image, img_size=128):
                """Preprocess the WIDERFACE image for model inference."""
                transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))  # Assuming single-channel grayscale
                ])
                image = Image.open(image).convert("RGB")  # Convert to RGB in case images are grayscale
                return transform(image).unsqueeze(0)  # Add batch dimension

            checkpoint_path = f"{MODELS_PATH}\\{MODEL_NAME}.ckpt"
            checkpoint = torch.load(checkpoint_path)

            if "criterion.pos_weight" in checkpoint["state_dict"]:
                del checkpoint["state_dict"]["criterion.pos_weight"]

            torch.save(checkpoint, checkpoint_path)

            model = FaceIDModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model.model,
                train_loader=None,
                val_loader=None 
            )

            criterion = nn.BCEWithLogitsLoss()
            model.criterion = criterion

            model.eval()
            model.freeze()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            processed_faces_folder = CSV_PATH + "\\processed_wider_faces"
            processed_faces = [os.path.join(processed_faces_folder, f) for f in os.listdir(processed_faces_folder) if f.endswith(".jpg")]

            annotations = pd.read_json(processed_faces_folder + "\\annotations_DONE.json")

            trait_test = trait[0]

            all_predictions = []
            all_targets     = []

            print(f"Number of pictures: {len(processed_faces)}")
            for face_idx, face_image_path in enumerate(processed_faces):
                preprocessed_face = preprocess_widerface(face_image_path).to(device)

                with torch.no_grad():
                    output = model(preprocessed_face)
                    label = 1.0 if output > 0.5 else 0.0
                    all_predictions.append(torch.tensor([label], dtype=torch.float32))

                    ground_truth = 1.0 if annotations["attributes"][face_idx][trait_test] else 0.0
                    all_targets.append(torch.tensor([ground_truth], dtype=torch.float32))

            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)

            test_accuracy = 0
            
            for pred_i, pred in enumerate(all_predictions):
                if all_targets[pred_i] == pred:
                    test_accuracy += 1
            
            print(f"Test Accuracy: {100*(test_accuracy/len(all_predictions)):.4f}")

            all_predictions_np = all_predictions.numpy().astype(int)
            all_targets_np = all_targets.numpy().astype(int)

            cm = confusion_matrix(all_targets_np, all_predictions_np)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix - {trait[0]}')
            plt.show()

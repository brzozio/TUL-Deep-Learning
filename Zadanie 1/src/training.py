import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

SRC_PATH:    str = os.path.dirname(os.path.abspath(__file__))
CSV_PATH:    str = os.path.dirname(SRC_PATH) + '\\csv'
JSON_PATH:   str = os.path.dirname(SRC_PATH) + '\\json'
MODELS_PATH: str = os.path.dirname(SRC_PATH) + '\\models'

TRAIN:          bool = False
CONTINUE_TRAIN: bool = False
MODEL_NAME:     str  = 'torch' # "nasz" albo "torch"

class FaceIDModel(pl.LightningModule):
    def __init__(self, model, train_loader, val_loader, criterion, lr=1e-4):
        super(FaceIDModel, self).__init__()
        self.model = model
        self.criterion = criterion
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
    
    transform = transforms.Compose([
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

    attr_df = attr_df[["image_id"] + traits]
    filtered_df = attr_df[(attr_df[traits] > 0).any(axis=1)]

    partition_df = pd.read_csv(partition_path, sep=' ', header=None, names=["image_id", "partition"])
    partition_df["image_id"] = partition_df["image_id"].astype(str)

    filtered_df = filtered_df.merge(partition_df, on="image_id")

    train_df = filtered_df[filtered_df["partition"] == 0].drop(columns=["partition"])
    val_df   = filtered_df[filtered_df["partition"] == 1].drop(columns=["partition"])
    test_df  = filtered_df[filtered_df["partition"] == 2].drop(columns=["partition"])

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One of the dataset splits is empty. Check your data files and partition file.")

    train_data = CelebADataset(img_folder_path, train_df, transform)
    val_data   = CelebADataset(img_folder_path, val_df, transform)
    test_data  = CelebADataset(img_folder_path, test_df, transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    pos_weight = torch.tensor([len(filtered_df) / (2 * filtered_df[traits].sum().sum())])

    return train_loader, val_loader, test_loader, pos_weight

def train(model):
    max_epochs = 100

    model.train()

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
        callbacks=[early_stopping, checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None
    )

    faceid_trainer.fit(model)


if __name__ == "__main__":
    batch_size = 32
    img_size = 128
    if MODEL_NAME == "nasz":
        model_pre    = FaceID_CNN()
        # criterion    = nn.BCELoss()
        train_loader, val_loader, test_loader, pos_weight = prepare_data(batch_size=batch_size, img_size=img_size, traits=['Male'])
        criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model        = FaceIDModel(model_pre, train_loader, val_loader, criterion)
    elif MODEL_NAME == "torch":
        model_pre = Ready_faceID_CNN()
        # criterion    = nn.BCELoss()
        train_loader, val_loader, test_loader, pos_weight = prepare_data(batch_size=batch_size, img_size=img_size, traits=['Eyeglasses'])
        criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model        = FaceIDModel(model_pre, train_loader, val_loader, criterion)

    if TRAIN:
        if CONTINUE_TRAIN:
            checkpoint_path = MODELS_PATH + f"\\{MODEL_NAME}.ckpt"
            model_continue = FaceIDModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model.model,
                train_loader=train_loader,
                val_loader=val_loader
            )
            # model.set_criterion(criterion=criterion)
            train(model=model_continue)
        else:
            # model.set_criterion(criterion=criterion)
            train(model=model)
    else:
        checkpoint_path = MODELS_PATH + f"\\{MODEL_NAME}.ckpt"

        model = FaceIDModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model.model,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # model.set_criterion(criterion=criterion)

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

        test_accuracy = (all_predictions == all_targets).float().mean().item()
        print(f"Test Accuracy: {test_accuracy:.4f}")
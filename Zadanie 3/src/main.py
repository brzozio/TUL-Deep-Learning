import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.datasets import MoleculeNet, QM9
from torchmetrics.classification import BinaryAccuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.nn import TransformerConv
from model import GNNModel

# Paths and Constants
SRC_PATH = os.path.abspath(os.path.dirname(__file__))
CSV_PATH = os.path.join(os.path.dirname(SRC_PATH), "csv")
MODELS_PATH = os.path.join(os.path.dirname(SRC_PATH), "models")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10

TRAIN = True
CONTINUE_TRAIN = False
# MODEL_NAME = "classification"
MODEL_NAME = "regression"

def plot_losses(train_losses, val_losses):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()

class FaceIDModel(pl.LightningModule):
    def __init__(self, model, train_loader, val_loader, task_type='classification', lr=1e-4):
        """
        task_type: 'classification' for classification tasks, 'regression' for regression tasks
        """
        super(FaceIDModel, self).__init__()
        self.model = model
        self.lr = lr
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_type = task_type

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        y_hat = self(batch)
        y = y.squeeze().long()

        # print(f"Training: {y.shape}, y={y}")
        # print(f"Training y_hat: {y_hat.shape}, y_hat={y_hat}")

        if self.task_type == 'classification':
            # y_hat = (y_hat > 0.5).long()
            # print(f"Training classification y_hat: {y_hat.shape}, y_hat={y_hat}")
            loss = nn.CrossEntropyLoss()(y_hat, y)
            
        elif self.task_type == 'regression':
            loss = nn.MSELoss()(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
    
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        y_hat = self(batch)
        y = y.squeeze().long()

        # print(f"Valid y: {y.shape}, y={y}")
        # print(f"Valid y_hat: {y_hat.shape}, y_hat={y_hat}")
        
        if self.task_type == 'classification':
            # y_hat = (y_hat > 0.5).long()
            # print(f"Valid classification y_hat: {y_hat.shape}, y_hat={y_hat}")
            loss = nn.CrossEntropyLoss()(y_hat, y)
            
        elif self.task_type == 'regression':
            loss = nn.MSELoss()(y_hat, y)
        
        self.log("val_loss", loss, prog_bar=True)
    
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

def get_data(dataset_name, task_type, filter_inhibitors=None):
   
    if task_type == 'classification':
        dataset = MoleculeNet(root=os.path.join(CSV_PATH, "dataset", dataset_name), name=dataset_name)
        
        if filter_inhibitors is not None:
            filtered_indices = [i for i, data in enumerate(dataset) if data.y.item() == filter_inhibitors]
            dataset = dataset[filtered_indices]

    elif task_type == 'regression':
        dataset = QM9(root=os.path.join(CSV_PATH, "dataset", dataset_name))

    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    test_idx, val_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    print(f"x type: {type(train_dataset[0].x)} | shape: {train_dataset[0].x.shape if isinstance(train_dataset[0].x, torch.Tensor) else 'Not a tensor'}")
    print(f"edge_index type: {type(train_dataset[0].edge_index)} | shape: {train_dataset[0].edge_index.shape if isinstance(train_dataset[0].edge_index, torch.Tensor) else 'Not a tensor'}")

    return (
        GeoDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8),
        GeoDataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=8),
        GeoDataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=8)
    )

def train_model(model):
    """Train the PyTorch Lightning model."""
    train_losses, val_losses = [], []

    class LossLogger(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            train_loss = trainer.callback_metrics.get("train_loss")
            if train_loss:
                train_losses.append(train_loss.item())

        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss:
                val_losses.append(val_loss.item())

    loss_logger = LossLogger()
    model.train()

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODELS_PATH,
        filename=f"{MODEL_NAME}_{{epoch:03d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=PATIENCE, verbose=True, mode="min")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint_callback, loss_logger],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None
    )

    trainer.fit(model)

    return train_losses, val_losses

if __name__ == "__main__":
    if MODEL_NAME == "classification":
        train_loader, val_loader, test_loader = get_data('BACE', 'classification', filter_inhibitors=True)
        is_classification = True
    elif MODEL_NAME == "regression":
        train_loader, val_loader, test_loader = get_data('QM9', 'regression')
        is_classification = False

    model_core = GNNModel(is_classification=is_classification).to(DEVICE)
    model = FaceIDModel(model_core, train_loader, val_loader, MODEL_NAME, lr=LEARNING_RATE)

    if TRAIN:
        if CONTINUE_TRAIN:
            checkpoint_path = os.path.join(MODELS_PATH, f"{MODEL_NAME}.ckpt")
            model = FaceIDModel.load_from_checkpoint(checkpoint_path, model=model_core)
        train_losses, val_losses = train_model(model)
        plot_losses(train_losses, val_losses)
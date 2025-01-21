import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.datasets import MoleculeNet, QM9
from torchmetrics.classification import BinaryAccuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.nn import TransformerConv
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import GNNModelRegression,  GNNModelClassification as GNNModel

# Paths and Constants
SRC_PATH = os.path.abspath(os.path.dirname(__file__))
CSV_PATH = os.path.join(os.path.dirname(SRC_PATH), "csv")
MODELS_PATH = os.path.join(os.path.dirname(SRC_PATH), "models")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10

TRAIN = False
CONTINUE_TRAIN = True
# MODEL_NAME = "classification"
MODEL_NAME = "regression"

class DipoleMomentTransform(BaseTransform):
            def __call__(self, data):
                # Select the dipole moment (index 4 in `data.y`)
                data.y = data.y[:, 4].view(-1, 1)  # Ensure shape [N, 1]
                return data

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

        # print(f"Training: {y.shape}, y={y}")
        # print(f"Training y_hat: {y_hat.shape}, y_hat={y_hat}")

        if self.task_type == 'classification':
            # y_hat = (y_hat > 0.5).long()
            # print(f"Training classification y_hat: {y_hat.shape}, y_hat={y_hat}")
            y = y.squeeze().long()
            loss = nn.CrossEntropyLoss()(y_hat, y)
            
        elif self.task_type == 'regression':
            y = y.squeeze().float()
            loss = nn.MSELoss()(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
    
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        y_hat = self(batch)

        # print(f"Valid y: {y.shape}, y={y}")
        # print(f"Valid y_hat: {y_hat.shape}, y_hat={y_hat}")
        
        if self.task_type == 'classification':
            # y_hat = (y_hat > 0.5).long()
            # print(f"Valid classification y_hat: {y_hat.shape}, y_hat={y_hat}")
            y = y.squeeze().long()
            loss = nn.CrossEntropyLoss()(y_hat, y)
            
        elif self.task_type == 'regression':
            y = y.squeeze().float()
            loss = nn.MSELoss()(y_hat, y)
        
        self.log("val_loss", loss, prog_bar=True)
    
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

def get_data(dataset_name, task_type):
   
    if task_type == 'classification':
        dataset = MoleculeNet(root=os.path.join(CSV_PATH, "dataset", dataset_name), name=dataset_name)

    elif task_type == 'regression':

        dataset = QM9(root='QM9', transform=DipoleMomentTransform())

    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    test_idx, val_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    # if task_type == "classification":
    #     filtered_indices = [i for i, data in enumerate(train_dataset) if data.y.item() == 1]
    #     train_dataset = train_dataset[filtered_indices]

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
        train_loader, val_loader, test_loader = get_data('BACE', 'classification')
        model_core = GNNModel().to(DEVICE)
        
    elif MODEL_NAME == "regression":
        train_loader, val_loader, test_loader = get_data('QM9', 'regression')
        model_core = GNNModelRegression().to(DEVICE)

    model = FaceIDModel(model_core, train_loader, val_loader, MODEL_NAME, lr=LEARNING_RATE)

    if TRAIN:
        if CONTINUE_TRAIN:
            checkpoint_path = os.path.join(MODELS_PATH, f"{MODEL_NAME}.ckpt")
            model = FaceIDModel.load_from_checkpoint(checkpoint_path, model=model_core, train_loader=train_loader, val_loader=val_loader)
        train_losses, val_losses = train_model(model)
        plot_losses(train_losses, val_losses)
    else:
        checkpoint_path = MODELS_PATH + f"\\{MODEL_NAME}.ckpt"

        model = FaceIDModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model.model,
            train_loader=train_loader,
            val_loader=val_loader
        )

        criterion = nn.BCEWithLogitsLoss() if MODEL_NAME == 'classification' else nn.MSELoss()
        model.criterion = criterion

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        model.eval()
        model.freeze()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch.x.to(device), batch.y.to(device)
                edge_index = batch.edge_index.to(device)

                outputs = model(batch.to(device))

                if MODEL_NAME == 'classification':
                    y = y.squeeze().long()
                    loss = nn.CrossEntropyLoss()(outputs, y)
                    probabilities = torch.softmax(outputs, dim=-1)
                    predictions = probabilities.argmax(dim=-1)

                elif MODEL_NAME == 'regression':
                    y = y.squeeze().float()
                    loss = nn.MSELoss()(outputs.squeeze(), y)
                    predictions = outputs.squeeze()

                all_predictions.append(predictions.cpu())
                all_targets.append(y.cpu())

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        if MODEL_NAME == 'classification':
            test_accuracy = (all_predictions == all_targets).float().mean().item() * 100
            print(f"Test Accuracy: {test_accuracy:.4f}%")

            cm = confusion_matrix(all_targets, all_predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.show()

        elif MODEL_NAME == 'regression':
            mse = torch.mean((all_predictions - all_targets) ** 2).item()
            print(f"Mean Squared Error (MSE): {mse:.4f}")
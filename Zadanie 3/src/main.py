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
from model import GNNModelRegression, GNNModelRegression_GCNConv,  GNNModelClassification, GNNModelClassification_GCNConv
import numpy as np

# Paths and Constants
SRC_PATH = os.path.abspath(os.path.dirname(__file__))
CSV_PATH = os.path.join(os.path.dirname(SRC_PATH), "csv")
MODELS_PATH = os.path.join(os.path.dirname(SRC_PATH), "models")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE:     int   = 128
LEARNING_RATE:  float = 1e-4
TRAIN:          bool = False
CONTINUE_TRAIN: bool = False

# MODEL_NAME: str = "classification"
MODEL_NAME: str  = "regression"

class DipoleMomentTransform(BaseTransform):
            def __call__(self, data):
                # Select the dipole moment (index 4 in `data.y`)
                data.y = data.y[:, 4].view(-1, 1)  # Ensure shape [N, 1]
                return data

def plot_losses(train_losses, val_losses, modelname):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(modelname)
    plt.legend()
    plt.grid()
    plt.savefig(CSV_PATH+f"\\Losses_{modelname}.jpg")

def plot_decision_boundary(X, func, modelname, y_true=None)-> None:
    try:
        fig, ax = plt.subplots()
        # Definiujemy zakres dla osi x i y
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # Tworzymy siatkę punktów w celu wygenerowania granicy decyzyjnej
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # Obliczamy etykiety dla każdego punktu w siatce
        meshgrid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.double)
        Z = func(meshgrid_tensor)
        Z = torch.argmax(Z,dim=1).detach().numpy().reshape(xx.shape)
        print(Z)

        #Predykcja etykiet dla danych testowych podanych w funkcji
        if y_true is None:
            y_tested = torch.argmax(func(X), dim=1).numpy()
        else: y_tested=y_true

        # Rysujemy kontury granicy decyzyjnej
        ax.contourf(xx, yy, Z, alpha=0.4)
        
        # Rysujemy punkty treningowe
        ax.scatter(X[:, 0], X[:, 1], c=y_tested, s=20, edgecolors='k')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(modelname)
        plt.savefig(CSV_PATH+f"\\Decision_Boundary_{modelname}.jpg")
    except Exception as e:
        print(f"Failed to do decision boundary for : {modelname} with an error: {e}")

class FaceIDModel(pl.LightningModule):
    def __init__(self, model, train_loader, val_loader, task_type, lr=1e-4):
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
        GeoDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, persistent_workers=True),
        GeoDataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=6, persistent_workers=True),
        GeoDataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=6)
    )

def train_model(model, modelname):
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
        filename=f"{modelname}_{{epoch:03d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    if "regression" in modelname:
        EPOCHS:         int   = 100
        PATIENCE:       int   = 3
    else:
        EPOCHS:         int   = 400
        PATIENCE:       int   = 10


    early_stopping = EarlyStopping(monitor="val_loss", patience=PATIENCE, verbose=True, mode="min")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint_callback, loss_logger],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None
    )

    trainer.fit(model)

    return train_losses, val_losses

def main_transformer_conv() -> None:

    EMBEDDINGS:     list  = [1,2,32]
    LINEAR_REGRESSION:   list  = [True, False]
    # MODEL_NAME:     list  = ["regression", "classification"]
    MODEL_NAME:     list  = ["classification"]

    for model_name_id in MODEL_NAME:
        loop_count: int = len(LINEAR_REGRESSION) if model_name_id == "regression" else 1
        for embed_id in EMBEDDINGS:

            for linear_id in range(loop_count):

                if model_name_id == "classification":
                    train_loader, val_loader, test_loader = get_data('BACE', 'classification')
                    model_core = GNNModelClassification(embedding=embed_id).to(DEVICE)
                    saveModelName: str = f"TransformerConv_{model_name_id}_{embed_id}"
                    
                elif model_name_id == "regression":
                    train_loader, val_loader, test_loader = get_data('QM9', 'regression')
                    model_core = GNNModelRegression(embedding=embed_id, linear=LINEAR_REGRESSION[linear_id]).to(DEVICE)
                    saveModelName: str = f"TransformerConv_{model_name_id}_{embed_id}_{'liniowy' if LINEAR_REGRESSION[linear_id] == True else 'nieliniowy'}"

                model = FaceIDModel(model=model_core, train_loader=train_loader, val_loader=val_loader, task_type=model_name_id, lr=LEARNING_RATE)

                if TRAIN:
                    print(f"Running a training for {saveModelName}")
                    if CONTINUE_TRAIN:
                        checkpoint_path = os.path.join(MODELS_PATH, f"{model_name_id}.ckpt")
                        model = FaceIDModel.load_from_checkpoint(checkpoint_path, model=model_core, train_loader=train_loader, val_loader=val_loader, task_type=model_name_id, lr=LEARNING_RATE)
                    train_losses, val_losses = train_model(model, saveModelName)
                    plot_losses(train_losses, val_losses, saveModelName)
                else:
                    checkpoint_path = MODELS_PATH + f"\\{saveModelName}.ckpt"

                    model = FaceIDModel.load_from_checkpoint(checkpoint_path, model=model_core, train_loader=train_loader, val_loader=val_loader, task_type=model_name_id, lr=LEARNING_RATE)

                    criterion = nn.BCEWithLogitsLoss() if model_name_id == 'classification' else nn.MSELoss()
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

                            if model_name_id == 'classification':
                                y = y.squeeze().long()
                                loss = nn.CrossEntropyLoss()(outputs, y)
                                probabilities = torch.softmax(outputs, dim=-1)
                                predictions = probabilities.argmax(dim=-1)

                            elif model_name_id == 'regression':
                                y = y.squeeze().float()
                                loss = nn.MSELoss()(outputs.squeeze(), y)
                                predictions = outputs.squeeze()

                            all_predictions.append(predictions.cpu())
                            all_targets.append(y.cpu())

                    all_predictions = torch.cat(all_predictions)
                    all_targets = torch.cat(all_targets)

                    if model_name_id == 'classification':
                        test_accuracy = (all_predictions == all_targets).float().mean().item() * 100
                        print(f"Test Accuracy: {test_accuracy:.4f}%")

                        cm = confusion_matrix(all_targets, all_predictions)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                        plt.xlabel('Predicted Labels')
                        plt.ylabel('True Labels')
                        plt.title(saveModelName)
                        plt.show()

                    elif model_name_id == 'regression':
                        mse = torch.mean((all_predictions - all_targets) ** 2).item()
                        print(f"Mean Squared Error (MSE) {saveModelName}: {mse:.4f}")

                model            = None
                train_loader     = None
                val_loader       = None
                saveModelName    = None
                test_loader      = None
                model_core       = None

def main_gcn_conv() -> None:

    EMBEDDINGS:     list  = [1,2,32]
    LINEAR_REGRESSION:   list  = [True, False]
    # MODEL_NAME:     list  = ["regression", "classification"]
    MODEL_NAME:     list  = ["classification"]

    for model_name_id in MODEL_NAME:
        loop_count: int = len(LINEAR_REGRESSION) if model_name_id == "regression" else 1
        for embed_id in EMBEDDINGS:

            for linear_id in range(loop_count):

                if model_name_id == "classification":
                    train_loader, val_loader, test_loader = get_data('BACE', 'classification')
                    model_core = GNNModelClassification_GCNConv(embedding=embed_id).to(DEVICE)
                    saveModelName: str = f"GCNConv_{model_name_id}_{embed_id}"
                    
                elif model_name_id == "regression":
                    train_loader, val_loader, test_loader = get_data('QM9', 'regression')
                    model_core = GNNModelRegression_GCNConv(embedding=embed_id, linear=LINEAR_REGRESSION[linear_id]).to(DEVICE)
                    saveModelName: str = f"GCNConv_{model_name_id}_{embed_id}_{'liniowy' if LINEAR_REGRESSION[linear_id] == True else 'nieliniowy'}"

                model = FaceIDModel(model=model_core, train_loader=train_loader, val_loader=val_loader, task_type=model_name_id, lr=LEARNING_RATE)

                if TRAIN:
                    print(f"Running a training for {saveModelName}")
                    if CONTINUE_TRAIN:
                        checkpoint_path = os.path.join(MODELS_PATH, f"{model_name_id}.ckpt")
                        model = FaceIDModel.load_from_checkpoint(checkpoint_path, model=model_core, train_loader=train_loader, val_loader=val_loader, task_type=model_name_id, lr=LEARNING_RATE)
                    train_losses, val_losses = train_model(model, saveModelName)
                    plot_losses(train_losses, val_losses, saveModelName)
                else:
                    checkpoint_path = MODELS_PATH + f"\\{saveModelName}.ckpt"

                    model = FaceIDModel.load_from_checkpoint(checkpoint_path, model=model_core, train_loader=train_loader, val_loader=val_loader, task_type=model_name_id, lr=LEARNING_RATE)

                    criterion = nn.BCEWithLogitsLoss() if model_name_id == 'classification' else nn.MSELoss()
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

                            if model_name_id == 'classification':
                                y = y.squeeze().long()
                                loss = nn.CrossEntropyLoss()(outputs, y)
                                probabilities = torch.softmax(outputs, dim=-1)
                                predictions = probabilities.argmax(dim=-1)

                            elif model_name_id == 'regression':
                                y = y.squeeze().float()
                                loss = nn.MSELoss()(outputs.squeeze(), y)
                                predictions = outputs.squeeze()

                            all_predictions.append(predictions.cpu())
                            all_targets.append(y.cpu())

                    all_predictions = torch.cat(all_predictions)
                    all_targets = torch.cat(all_targets)

                    if model_name_id == 'classification':
                        test_accuracy = (all_predictions == all_targets).float().mean().item() * 100
                        print(f"Test Accuracy: {test_accuracy:.4f}%")

                        cm = confusion_matrix(all_targets, all_predictions)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                        plt.xlabel('Predicted Labels')
                        plt.ylabel('True Labels')
                        plt.title(saveModelName)
                        plt.show()

                    elif model_name_id == 'regression':
                        mse = torch.mean((all_predictions - all_targets) ** 2).item()
                        print(f"Mean Squared Error (MSE) {saveModelName}: {mse:.4f}")

                model            = None
                train_loader     = None
                val_loader       = None
                saveModelName    = None
                test_loader      = None
                model_core       = None

if __name__ == "__main__":
    
   main_transformer_conv()
   main_gcn_conv()
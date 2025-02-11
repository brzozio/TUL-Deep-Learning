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
TRAIN:          bool = True
CONTINUE_TRAIN: bool = False

log: list = []

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

def generate_decision_boundary(test_loader, model, embedding_size, model_name):
    """
    Generate a decision boundary plot for classification tasks.
    Args:
        test_loader: DataLoader for the test set.
        model: Trained classification model.
        embedding_size: Size of the embedding space.
        model_name: Name used for plot title and filename.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch.x.to(DEVICE).float()
            edge_index = batch.edge_index.to(DEVICE).long()
            batch_labels = batch.y.to(DEVICE)
            batch_vector = batch.batch.to(DEVICE)

            print(f"X type: {x.dtype}, edge_index type: {edge_index.dtype}")

            # Forward pass to extract embeddings
            embeddings = model.model.conv1(x, edge_index)
            embeddings = model.model.conv2(embeddings, edge_index)
            embeddings = model.model.pool(embeddings, batch_vector)

            all_features.append(embeddings.cpu())
            all_labels.append(batch_labels.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Reduce to 2D if embeddings > 2 dimensions
    if embedding_size > 2:
        pca = PCA(n_components=2)
        all_features = pca.fit_transform(all_features)

    # 1D case: simple scatter plot along x-axis
    if all_features.shape[1] == 1:
        plt.figure(figsize=(8, 6))
        plt.scatter(all_features[:, 0], np.zeros_like(all_features[:, 0]),
                    c=all_labels, cmap='coolwarm', s=20, edgecolor='k')
        plt.colorbar(label="Class")
        plt.title(f"{model_name} (Embedding Size={embedding_size})")
        plt.xlabel("Feature 1")
        plt.grid(True)
        plt.savefig(CSV_PATH + f"/Decision_Boundary_{model_name}_{embedding_size}.jpg")
        plt.show()
    else:
        # Create a tighter grid: smaller margin and higher resolution
        margin = 0.1
        x_min, x_max = all_features[:, 0].min() - margin, all_features[:, 0].max() + margin
        y_min, y_max = all_features[:, 1].min() - margin, all_features[:, 1].max() + margin
        resolution = 300  # increase resolution for a denser grid
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            # Assumes model's forward can accept grid_points_tensor (2D inputs) and return logits.
            grid_predictions = model(grid_points_tensor)
            grid_predictions = torch.softmax(grid_predictions, dim=-1).argmax(dim=-1).cpu().numpy()

        grid_predictions = grid_predictions.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, grid_predictions, alpha=0.6, cmap='coolwarm')
        plt.scatter(all_features[:, 0], all_features[:, 1], c=all_labels,
                    edgecolor='k', cmap='coolwarm', s=20)
        plt.title(f"{model_name} (Embedding Size={embedding_size})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="Class")
        plt.savefig(CSV_PATH + f"/Decision_Boundary_{model_name}_{embedding_size}.jpg")
        plt.show()

def visualize_embedding_space(model, data_loader, task_type, embedding_size, model_name, csv_path):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            x, edge_index, batch_labels = batch.x.to(DEVICE), batch.edge_index.to(DEVICE), batch.y.to(DEVICE)

            x = x.float()
            edge_index = edge_index.long()

            embeddings = model.model.conv1(x, edge_index)
            embeddings = model.model.conv2(embeddings, edge_index)
            embeddings = model.model.pool(embeddings, batch.batch.to(DEVICE))

            all_embeddings.append(embeddings.cpu())
            all_labels.append(batch_labels.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    if embedding_size > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        all_embeddings = pca.fit_transform(all_embeddings)

    plt.figure(figsize=(8, 6))
    
    if task_type == "regression":
        scatter = plt.scatter(
            all_embeddings[:, 0], all_embeddings[:, 1], 
            c=all_labels.squeeze(), cmap="viridis", s=20, edgecolor="k", alpha=0.75
        )
        plt.colorbar(scatter, label="Predicted Value")
    else:
        scatter = plt.scatter(
            all_embeddings[:, 0], all_embeddings[:, 1], 
            c=all_labels.squeeze(), cmap="coolwarm", s=20, edgecolor="k"
        )
        plt.colorbar(scatter, label="Class")

    plt.title(f"{model_name} (Embedding Size={embedding_size})")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.grid()
    plt.savefig(f"{csv_path}/Embedding_Space_{model_name}.jpg")
    plt.show()

# def visualize_embedding_space(model, data_loader, task_type, embedding_size, linear_type, model_name):
#     """
#     Visualizes the embedding space for a trained model.
#     """
#     model.eval()
#     all_embeddings = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in data_loader:
#             x, edge_index, batch_labels = batch.x.to(DEVICE), batch.edge_index.to(DEVICE), batch.y.to(DEVICE)

#             # Forward pass to extract embeddings
#             x = x.float()
#             edge_index = edge_index.long()

#             print(f"X type: {x.dtype}, edge_index type: {edge_index.dtype}")

#             embeddings = model.model.conv1(x, edge_index)
#             embeddings = model.model.conv2(embeddings, edge_index)
#             embeddings = model.model.pool(embeddings, batch.batch.to(DEVICE))

#             all_embeddings.append(embeddings.cpu())
#             all_labels.append(batch_labels.cpu())

#     # Combine embeddings and labels
#     all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
#     all_labels = torch.cat(all_labels, dim=0).numpy()

#     # If embedding space > 2 dimensions, reduce using PCA
#     if embedding_size > 2:
#         from sklearn.decomposition import PCA
#         pca = PCA(n_components=2)
#         all_embeddings = pca.fit_transform(all_embeddings)

#     # Check if embeddings are 1D or 2D
#     if all_embeddings.shape[1] == 1:
#         # If the embeddings are 1D, plot as a 1D scatter plot
#         plt.figure(figsize=(8, 6))
#         plt.scatter(all_embeddings[:, 0], np.zeros_like(all_embeddings[:, 0]), c=all_labels.squeeze(), cmap='coolwarm', s=20, edgecolor='k')
#         plt.colorbar(label="Class" if task_type == 'classification' else "Predicted Value")
#         plt.title(f"{model_name} (size={embedding_size})")
#         plt.xlabel("Embedding Dimension 1")
#         plt.grid()
#         plt.savefig(CSV_PATH + f"\\Embedding_Space_{model_name}.jpg")
#         plt.show()

#     else:
#         # If the embeddings are 2D or higher, plot normally
#         plt.figure(figsize=(8, 6))
#         scatter = plt.scatter(all_embeddings[:, 0], all_embeddings[:, 1], c=all_labels.squeeze(), cmap='coolwarm', s=20, edgecolor='k')
#         plt.colorbar(scatter, label="Class" if task_type == 'classification' else "Predicted Value")
#         plt.title(f"{model_name} (size={embedding_size})")
#         plt.xlabel("Embedding Dimension 1")
#         plt.ylabel("Embedding Dimension 2")
#         plt.grid()
#         plt.savefig(CSV_PATH + f"\\Embedding_Space_{model_name}.jpg")
#         plt.show()

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
    # MODEL_NAME:     list  = ["classification", "regression"]
    MODEL_NAME:     list  = ["regression"]

    for model_name_id in MODEL_NAME:
        # loop_count: int = len(LINEAR_REGRESSION) if model_name_id == "regression" else 1
        loop_count: int = len(LINEAR_REGRESSION)
        for embed_id in EMBEDDINGS:

            for linear_id in range(loop_count):

                if model_name_id == "classification":
                    train_loader, val_loader, test_loader = get_data('BACE', 'classification')
                    model_core = GNNModelClassification(embedding=embed_id, linear=LINEAR_REGRESSION[linear_id]).to(DEVICE)
                    saveModelName: str = f"TransformerConv_{model_name_id}_{embed_id}_{'liniowy' if LINEAR_REGRESSION[linear_id] == True else 'nieliniowy'}"
                    
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

                    # Plot decision boundary for classification
                    if model_name_id == 'classification':
                        generate_decision_boundary(test_loader, model, embed_id, saveModelName)
                    else:
                        # Visualize embedding space
                        visualize_embedding_space(
                            model=model,
                            data_loader=test_loader,
                            task_type=model_name_id,
                            embedding_size=embed_id,
                            linear_type=LINEAR_REGRESSION[linear_id],
                            model_name=saveModelName
                        )

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

                        log.append(f"Model: {saveModelName}, MSE: {mse:.4f}\n")

                model            = None
                train_loader     = None
                val_loader       = None
                saveModelName    = None
                test_loader      = None
                model_core       = None

def main_gcn_conv() -> None:

    EMBEDDINGS:     list  = [2,1,32]
    LINEAR_REGRESSION:   list  = [True, False]
    MODEL_NAME:     list  = ["classification","regression"]
    # MODEL_NAME:     list  = ["regression"]

    for model_name_id in MODEL_NAME:
        loop_count: int = len(LINEAR_REGRESSION)
        # loop_count: int = len(LINEAR_REGRESSION) if model_name_id == "regression" else 1
        for embed_id in EMBEDDINGS:

            for linear_id in range(loop_count):

                if model_name_id == "classification":
                    train_loader, val_loader, test_loader = get_data('BACE', 'classification')
                    model_core = GNNModelClassification_GCNConv(embedding=embed_id, linear=LINEAR_REGRESSION[linear_id]).to(DEVICE)
                    saveModelName: str = f"GCNConv_{model_name_id}_{embed_id}_{'liniowy' if LINEAR_REGRESSION[linear_id] == True else 'nieliniowy'}"
                    
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

                    # Plot decision boundary for classification
                    if model_name_id == 'classification':
                        generate_decision_boundary(test_loader, model, embed_id, saveModelName)
                    else:
                        # Visualize embedding space
                        visualize_embedding_space(
                            model=model,
                            data_loader=test_loader,
                            task_type=model_name_id,
                            embedding_size=embed_id,
                            linear_type=LINEAR_REGRESSION[linear_id],
                            model_name=saveModelName
                        )

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

                        log_file_path = os.path.join(CSV_PATH, "mse_log.txt")

                        if not os.path.exists(CSV_PATH):
                            print(f"Error: The directory {CSV_PATH} does not exist.")
                        else:
                            print(f"Logging MSE to: {log_file_path}")

                        try:
                            with open(log_file_path, "a") as log_file:
                                log_file.write(f"Model: {saveModelName}, MSE: {mse:.4f}\n")
                        except Exception as e:
                            print(f"Error writing to log file: {e}")

                        log.append(f"Model: {saveModelName}, MSE: {mse:.4f}\n")

                model            = None
                train_loader     = None
                val_loader       = None
                saveModelName    = None
                test_loader      = None
                model_core       = None

if __name__ == "__main__":
    
   main_gcn_conv()
   main_transformer_conv()

   print(log)
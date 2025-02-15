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
from matplotlib.colors import ListedColormap, BoundaryNorm

# Paths and Constants
SRC_PATH = os.path.abspath(os.path.dirname(__file__))
CSV_PATH = os.path.join(os.path.dirname(SRC_PATH), "csv")
MODELS_PATH = os.path.join(os.path.dirname(SRC_PATH), "models")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE:     int   = 64
LEARNING_RATE:  float = 1e-10
TRAIN:          bool = True
CONTINUE_TRAIN: bool = True

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
        GeoDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, persistent_workers=True),
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
        EPOCHS:         int   = 1000
        PATIENCE:       int   = 10
    else:
        EPOCHS:         int   = 10000
        PATIENCE:       int   = 200


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

    EMBEDDINGS:     list  = [1,2]
    LINEAR_REGRESSION:   list  = [True, False]
    MODEL_NAME:     list  = ["classification"]#, "regression"]

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
                        checkpoint_path = os.path.join(MODELS_PATH, f"{saveModelName}.ckpt")
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

                    # # Plot decision boundary for classification
                    # if model_name_id == 'classification':
                    #     generate_decision_boundary(test_loader, model, embed_id, saveModelName)
                    # else:
                    #     plot_regression_function(test_loader, model, embed_id, saveModelName)

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

                        # cm = confusion_matrix(all_targets, all_predictions)
                        # plt.figure(figsize=(8, 6))
                        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                        # plt.xlabel('Predicted Labels')
                        # plt.ylabel('True Labels')
                        # plt.title(saveModelName)
                        # plt.show()

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

def generate_decision_boundary(test_loader, model, embedding_size, model_name):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch.x.to(DEVICE).float()
            edge_index = batch.edge_index.to(DEVICE).long()
            batch_labels = batch.y.to(DEVICE)
            batch_vector = batch.batch.to(DEVICE)

            emb = model.model.conv1(x, edge_index)
            emb = model.model.conv2(emb, edge_index)
            emb = model.model.pool(emb, batch_vector)

            all_features.append(emb.cpu())
            all_labels.append(batch_labels.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_labels = np.squeeze(all_labels)  # usuwa nadmiarowe wymiary

    # Redukcja wymiaru, jeśli embedding > 2
    if embedding_size > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        all_features = pca.fit_transform(all_features)

    # 1D embedding
    if all_features.shape[1] == 1:
        # Ustalamy zakres osi x
        x_min, x_max = all_features[:, 0].min() - 1.0, all_features[:, 0].max() + 1.0
        grid_x = np.linspace(x_min, x_max, 300)
        grid_tensor = torch.tensor(grid_x, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            fc1 = model.fc1 if hasattr(model, "fc1") else model.model.fc1
            fc2 = model.fc2 if hasattr(model, "fc2") else getattr(model.model, "fc2", None)
            relu = model.relu if hasattr(model, "relu") else model.model.relu

            out = relu(fc1(grid_tensor))
            if fc2 is not None:
                out = fc2(out)
            out = out

            # Obsługa różnych wariantów wyjścia modelu:
            if out.shape[1] == 1:
                # Jeśli pojedynczy neuron (zakładamy, że już przez Sigmoid -> [0,1])
                probs = out.squeeze(1).cpu().numpy()
            else:
                # Jeśli dwa neurony, stosujemy softmax i wybieramy prawdopodobieństwo klasy 1.
                probs = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()

        plt.figure(figsize=(8, 6))
        # Rozdzielamy punkty: klasa 0 na y=0, klasa 1 na y=1
        x_class0 = all_features[all_labels == 0, 0]
        x_class1 = all_features[all_labels == 1, 0]
        # plt.scatter(x_class0, np.zeros_like(x_class0),
        #             color='blue', edgecolor='k', label='Klasa 0')
        # plt.scatter(x_class1, np.ones_like(x_class1),
        #             color='red', edgecolor='k', label='Klasa 1')
        
        plt.scatter(x_class0, np.zeros_like(x_class0), color='blue', label='Nie')
        plt.scatter(x_class1, np.zeros_like(x_class1), color='red', label='Inhibitor')
        
        # Rysujemy krzywą prawdopodobieństwa klasy 1
        plt.plot(grid_x, probs, color='green', label='Prawdopodobieństwo Inhibitor')
        # Szukamy punktów, gdzie p przekracza 0.5 (granica decyzyjna)
        boundary_indices = np.where(np.diff((probs >= 0.5).astype(int)) != 0)[0]
        for idx in boundary_indices:
            bx = (grid_x[idx] + grid_x[idx+1]) / 2.0
            plt.axvline(x=bx, color='gray', linestyle='--', alpha=0.7)

        plt.title(f"{model_name} (Embedding Size={embedding_size})")
        plt.xlabel("Feature 1")
        plt.ylabel("Prawdopodobieństwo / Klasa")
        plt.ylim([-0.2, 1.2])
        plt.grid(True)
        plt.legend()
        plt.savefig(CSV_PATH + f"/Decision_Boundary_{model_name}_{embedding_size}.jpg")
    else:
        # Generujemy siatkę
        margin = 1.0
        x_min, x_max = all_features[:,0].min() - margin, all_features[:,0].max() + margin
        y_min, y_max = all_features[:,1].min() - margin, all_features[:,1].max() + margin
        resolution = 300
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            # Wywołanie warstw FC modelu
            fc1 = model.fc1 if hasattr(model, "fc1") else model.model.fc1
            fc2 = model.fc2 if hasattr(model, "fc2") else getattr(model.model, "fc2", None)
            relu = model.relu if hasattr(model, "relu") else model.model.relu

            out = relu(fc1(grid_tensor))
            if fc2 is not None:
                out = fc2(out)
            out = out

            # Sprawdzamy wymiar wyjścia
            if out.shape[1] == 1:
                # Binary classification z pojedynczym wyjściem (Sigmoid)
                preds = (out >= 0.5).long().squeeze(dim=-1)
            else:
                # Binary classification (2 wyjścia) lub multi-class
                preds = torch.softmax(out, dim=-1).argmax(dim=-1)

            preds = preds.cpu().numpy()

        preds = preds.reshape(xx.shape)

        # Dyskretna mapa barw
        cmap = ListedColormap(['blue', 'red'])
        bounds = [-0.5, 0.5, 1.5]  # granice między klasami 0 i 1
        norm = BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(8,6))
        plt.contourf(xx, yy, preds, alpha=0.6, cmap=cmap, norm=norm)
        plt.scatter(all_features[:,0], all_features[:,1],
                    c=all_labels, cmap=cmap, norm=norm,
                    edgecolor='k', s=20)
        plt.title(f"{model_name} (Embedding Size={embedding_size})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(ticks=[0,1], label="Class")
        plt.savefig(CSV_PATH + f"/Decision_Boundary_{model_name}_{embedding_size}.jpg")


def plot_regression_function(test_loader, model, embedding_size, model_name):
    """
    Wizualizuje funkcję regresji.
    Dla 1D: wykres 2D (osadzenie vs wartość).
    Dla 2D: wykres 3D (powierzchnia funkcji).
    """
    model.eval()
    all_features = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch.x.to(DEVICE).float()
            edge_index = batch.edge_index.to(DEVICE).long()
            targets = batch.y.to(DEVICE)
            batch_vector = batch.batch.to(DEVICE)
            emb = model.model.conv1(x, edge_index)
            emb = model.model.conv2(emb, edge_index)
            emb = model.model.pool(emb, batch_vector)
            all_features.append(emb.cpu())
            all_targets.append(targets.cpu())
    all_features = torch.cat(all_features, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    if embedding_size > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        all_features = pca.fit_transform(all_features)
    if all_features.shape[1] == 1:
        margin = 0.1
        x_min, x_max = all_features[:,0].min()-margin, all_features[:,0].max()+margin
        grid_x = np.linspace(x_min, x_max, 300)
        grid_tensor = torch.tensor(grid_x, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            fc1 = model.fc1 if hasattr(model, "fc1") else model.model.fc1
            fc2 = model.fc2 if hasattr(model, "fc2") else getattr(model.model, "fc2", None)
            relu = model.relu if hasattr(model, "relu") else model.model.relu
            output_activation = model.output_activation if hasattr(model, "output_activation") else model.model.output_activation
            out = relu(fc1(grid_tensor))
            if fc2 is not None:
                out = fc2(out)
            out = output_activation(out)
            preds = out.squeeze().cpu().numpy()
        plt.figure(figsize=(8,6))
        plt.scatter(all_features[:,0], all_targets, color='blue', alpha=0.6, label='Test data')
        plt.plot(grid_x, preds, color='red', linewidth=2, label='Regression function')
        plt.title(f"{model_name} Regression (Embedding Size=1)")
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.legend()
        plt.grid(True)
        plt.savefig(CSV_PATH + f"/Regression_Function_{model_name}_1D.jpg")
    elif all_features.shape[1] == 2:
        margin = 0.1
        x_min, x_max = all_features[:,0].min()-margin, all_features[:,0].max()+margin
        y_min, y_max = all_features[:,1].min()-margin, all_features[:,1].max()+margin
        resolution = 100
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            fc1 = model.fc1 if hasattr(model, "fc1") else model.model.fc1
            fc2 = model.fc2 if hasattr(model, "fc2") else getattr(model.model, "fc2", None)
            relu = model.relu if hasattr(model, "relu") else model.model.relu
            output_activation = model.output_activation if hasattr(model, "output_activation") else model.model.output_activation
            out = relu(fc1(grid_tensor))
            if fc2 is not None:
                out = fc2(out)
            out = output_activation(out)
            preds = out.squeeze().cpu().numpy()
        preds = preds.reshape(xx.shape)
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, preds, cmap='viridis', alpha=0.8)
        ax.scatter(all_features[:,0], all_features[:,1], all_targets, color='red', s=20, label='Test data')
        ax.set_title(f"{model_name} Regression (Embedding Size=2)")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Target")
        plt.legend()
        plt.savefig(CSV_PATH + f"/Regression_Function_{model_name}_2D.jpg")

if __name__ == "__main__":
    
#    main_gcn_conv()
   main_transformer_conv()

   print(log)
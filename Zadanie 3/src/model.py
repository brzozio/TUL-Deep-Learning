import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool


class GNNModelClassification(nn.Module):
    def __init__(self, linear, embedding: int = 32, dropout_rate: float = 0.5):
        super(GNNModelClassification, self).__init__()

        self.conv1 = TransformerConv(in_channels=9, out_channels=16)
        self.conv2 = TransformerConv(in_channels=16, out_channels=embedding)

        self.pool = global_mean_pool

        if linear:
            self.fc1 = nn.Linear(embedding, 2)
            self.fc2 = None
        else:
            self.fc1 = nn.Linear(embedding, embedding * 2)
            self.fc2 = nn.Linear(embedding * 2, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        print(f"x type: {type(x)} | shape: {x.shape if isinstance(x, torch.Tensor) else 'Not a tensor'}")
        
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input x should be a tensor")
        
        if len(x.shape) != 2 or x.shape[1] != 9:
            raise ValueError(f"Unexpected shape of x. Expected [num_nodes, 9] but got {x.shape}")

        x = x.float()

        if edge_index is None or batch is None:
            raise ValueError("Input data must contain edge_index and batch.")

        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)
        x = self.dropout(x)

        if self.fc2 is None:
            x = self.relu(self.fc1(x))
        else:
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

        return x

class GNNModelRegression(nn.Module):
    def __init__(self, linear, embedding: int = 32):
        super(GNNModelRegression, self).__init__()

        self.conv1 = TransformerConv(in_channels=11, out_channels=16)
        self.conv2 = TransformerConv(in_channels=16, out_channels=embedding) #Dodatkowo 1 i 2 na out_channels

        self.pool = global_mean_pool

        if linear:
            self.fc1 = nn.Linear(embedding, 1)
            self.fc2 = None
        else:
            self.fc1 = nn.Linear(embedding, embedding * 2)
            self.fc2 = nn.Linear(embedding * 2, 1)

        self.relu = nn.ReLU()
        self.output_activation = nn.Identity()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        print(f"x type: {type(x)} | shape: {x.shape if isinstance(x, torch.Tensor) else 'Not a tensor'}")
        
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input x should be a tensor")
        
        if len(x.shape) != 2 or x.shape[1] != 11:
            raise ValueError(f"Unexpected shape of x. Expected [num_nodes, 11] but got {x.shape}")

        x = x.float()

        if edge_index is None or batch is None:
            raise ValueError("Input data must contain edge_index and batch.")

        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))

        x = self.pool(x, batch)

        if self.fc2 is None:
            x = self.relu(self.fc1(x))
        else:
            x = self.relu(self.fc1(x))
            x = self.fc2(x)

        return self.output_activation(x)
    
class GNNModelClassification_GCNConv(nn.Module):
    def __init__(self, linear,  embedding: int = 32):
        super(GNNModelClassification_GCNConv, self).__init__()

        self.conv1 = GCNConv(in_channels=9, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=embedding)

        self.pool = global_mean_pool

        if linear:
            self.fc1 = nn.Linear(embedding, 2)
            self.fc2 = None
        else:
            self.fc1 = nn.Linear(embedding, embedding * 2)
            self.fc2 = nn.Linear(embedding * 2, 2)

        # self.fc1 = nn.Linear(embedding, embedding*2)
        # self.fc2 = nn.Linear(embedding*2, 2)

        self.relu = nn.ReLU()
        # self.output_activation = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        print(f"x type: {type(x)} | shape: {x.shape if isinstance(x, torch.Tensor) else 'Not a tensor'}")
        
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input x should be a tensor")
        
        if len(x.shape) != 2 or x.shape[1] != 9:
            raise ValueError(f"Unexpected shape of x. Expected [num_nodes, 9] but got {x.shape}")

        x = x.float()

        if edge_index is None or batch is None:
            raise ValueError("Input data must contain edge_index and batch.")

        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))

        x = self.pool(x, batch)

        if self.fc2 is None:
            x = self.relu(self.fc1(x))
        else:
            x = self.relu(self.fc1(x))
            x = self.fc2(x)

        return x

class GNNModelRegression_GCNConv(nn.Module):
    def __init__(self, linear, embedding: int = 32):
        super(GNNModelRegression_GCNConv, self).__init__()

        self.conv1 = GCNConv(in_channels=11, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=embedding) #Dodatkowo 1 i 2 na out_channels

        self.pool = global_mean_pool

        if linear:
            self.fc1 = nn.Linear(embedding, 1)
            self.fc2 = None
        else:
            self.fc1 = nn.Linear(embedding, embedding * 2)
            self.fc2 = nn.Linear(embedding * 2, 1)

        self.relu = nn.ReLU()
        self.output_activation = nn.Identity()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        print(f"x type: {type(x)} | shape: {x.shape if isinstance(x, torch.Tensor) else 'Not a tensor'}")
        
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input x should be a tensor")
        
        if len(x.shape) != 2 or x.shape[1] != 11:
            raise ValueError(f"Unexpected shape of x. Expected [num_nodes, 11] but got {x.shape}")

        x = x.float()

        if edge_index is None or batch is None:
            raise ValueError("Input data must contain edge_index and batch.")

        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))

        x = self.pool(x, batch)

        if self.fc2 is None:
            x = self.relu(self.fc1(x))
        else:
            x = self.relu(self.fc1(x))
            x = self.fc2(x)

        return self.output_activation(x)
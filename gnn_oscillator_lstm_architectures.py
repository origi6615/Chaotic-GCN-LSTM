import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.nn import Parameter, BatchNorm1d
import pandas as pd
import numpy as np
class EC_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edge_types=1):
        super(EC_GCNConv, self).__init__(aggr='max')  # "Max" aggregation
        self.weights = Parameter(torch.Tensor(num_edge_types, out_channels, in_channels))
        self.bias = Parameter(torch.Tensor(out_channels))  # Added bias parameter
        self.weights.data.normal_(0, 0.001)
        self.bias.data.fill_(0)  # Initializing bias with zeros
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types

    def forward(self, x, edge_index, edge_type):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for i in range(self.num_edge_types):
            edge_mask = edge_type == i
            temp_edges = edge_index[:, edge_mask]
            out += F.linear(self.propagate(temp_edges, x=x, size=(x.size(0), x.size(0))), self.weights[i], bias=self.bias)
        return out

class NEC_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edge_types=1):
        super(NEC_GCNConv, self).__init__(aggr='max')  # "Max" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=True)  # Ensured bias is true by default

    def forward(self, x, edge_index, edge_type):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

class GNN(torch.nn.Module):
    def __init__(self, feature_dimension, edge_colours=False, num_edge_types=1,type_id=None):
        super(GNN, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=feature_dimension, hidden_size=feature_dimension)
        self.type_id = type_id
        if edge_colours:
            self.conv1 = EC_GCNConv(feature_dimension, 2 * feature_dimension, num_edge_types)
            self.conv2 = EC_GCNConv(2 * feature_dimension, feature_dimension, num_edge_types)
        else:
            self.conv1 = NEC_GCNConv(feature_dimension, 2 * feature_dimension, num_edge_types)
            self.conv2 = NEC_GCNConv(2 * feature_dimension, feature_dimension, num_edge_types)

        self.bn1 = BatchNorm1d(2 * feature_dimension)  # Added batch norm layer
        self.bn2 = BatchNorm1d(feature_dimension)  # Added batch norm layer
        
        self.lin_self_1 = torch.nn.Linear(feature_dimension, 2 * feature_dimension)
        self.lin_self_2 = torch.nn.Linear(2 * feature_dimension, feature_dimension)
        
        self.output = torch.nn.Sigmoid()
    
    def Sig_pie(self, k):
        s = 1
        return 1 / (1 + torch.exp(-s * k))
        
    def Oscillator(self, input):
        oscillator = pd.read_table("oscillators/Sig'/Type_%s.txt"%self.type_id, header=None)[0].tolist()
        inputs = torch.where(torch.logical_or(input < -0.5, input > 0.5), torch.FloatTensor([0.5001]).to(input.device), input)
        flat = torch.flatten(inputs)
        flat = torch.gather(torch.tensor(oscillator).to(input.device), 0, torch.floor(((flat + 0.5) * 10000)).to(torch.int64))
        inputs = torch.reshape(flat, inputs.shape)
        outputs = torch.where(inputs == 0, self.Sig_pie(input), inputs)
        return outputs



    
    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        
        
        x = self.Oscillator(x)
        
    
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        
        
        x = self.lin_self_1(x) + self.conv1(x, edge_index, edge_type)
        x = self.bn1(x)  
        x = torch.relu(x)
        x = self.lin_self_2(x) + self.conv2(x, edge_index, edge_type)
        x = self.bn2(x)  
        
        return self.output(x - 10)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_experts, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ) for _ in range(num_experts)])
        self.gates = nn.Sequential(
            nn.Linear(input_size, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        gates = self.gates(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
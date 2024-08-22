import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        torch.manual_seed(1)
        # Define layers based on input, hidden, and output sizes
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            # Initialize weights and biases randomly
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            nn.init.normal_(layer.bias, mean=0.0, std=1.0)
            self.layers.append(layer)

    def forward(self, x):
        """Compute the output with given input."""
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:  # Apply ReLU activation for all but the last layer
                    x = torch.relu(x)
        return x

    def get_num_parameters(self):
        """Get the number of parameters (weights and biases) in the MLP."""
        return sum(p.numel() for p in self.parameters())

    def get_parameters(self):
        """Get the parameters (weights and biases) of the MLP in a 1D vector."""
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        return torch.cat(params)

    def set_parameters(self, param_vector):
        """Set the MLP with given weights and biases in a 1D vector."""
        start = 0
        for layer in self.layers:
            weight_numel = layer.weight.numel()
            bias_numel = layer.bias.numel()

            layer.weight.data = param_vector[start:start + weight_numel].view_as(layer.weight).clone()
            start += weight_numel

            layer.bias.data = param_vector[start:start + bias_numel].view_as(layer.bias).clone()
            start += bias_numel

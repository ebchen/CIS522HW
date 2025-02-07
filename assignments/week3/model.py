import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    My implementatoin of multi-layer perceptron.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.actv = activation()
        # include dropout layers
        for i in range(hidden_count):
            layer = torch.nn.Linear(input_size, hidden_size)
            initializer(layer.weight)
            self.layers += [layer]
            self.layers += [torch.nn.BatchNorm1d(hidden_size)]
            if i % 2 == 0:
                self.layers += [torch.nn.ReLU()]
            else:
                self.layers += [torch.nn.Sigmoid()]
            self.layers += [torch.nn.Dropout(0.5)]
            input_size = hidden_size

        self.out = torch.nn.Linear(hidden_size, num_classes)
        initializer(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = self.actv(layer(x))
        x = self.out(x)
        return x

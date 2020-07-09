# -*- coding: utf-8 -*-
"""architectures.py

Author -- Mathias Wöß
Contact -- k11709064@students.jku.at
"""

import torch


class DeCropCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super(DeCropCNN, self).__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size,
                                       bias=True, padding=int(kernel_size / 2)))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        self.output_layer = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=1,
                                            kernel_size=kernel_size, bias=True, padding=int(kernel_size / 2))

    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        # outputs is the output of the CNN for each sample in the minibatch of size (n_samples, 1, X, Y).
        # Convert crop_array to boolean mask:
        crop_array = x[:, -1, ...]
        target_masks = crop_array.to(dtype=torch.bool)

        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        # remove second dimension (N, 1, X, Y) -> (N, X, Y)
        pred = torch.reshape(pred, (pred.shape[0], pred.shape[2], pred.shape[3]))
        # Use boolean mask as indices for each sample:
        predictions = [pred[i, target_masks[i]] for i in range(len(pred))]
        # predictions is now a list of n_samples elements, where each element corresponds to one sample.
        # Each element is a flattened tensor of shape (crop_size[0]*crop_size[1],), containing only the CNN outputs
        # at the cropped-out image parts that were indicated by crop_array.
        return predictions

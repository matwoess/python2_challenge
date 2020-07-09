# -*- coding: utf-8 -*-
"""main.py

Author -- Mathias Wöß
Contact -- k11709064@students.jku.at
"""

import os
import numpy as np
import torch
import torch.utils.data
from datasets import GreyscaleDataset, CroppedImages
from utils import plot
from architectures import DeCropCNN
import torchvision.transforms as transforms
# TODO: requirements.txt
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import tqdm


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`"""
    # Define a loss (mse loss)
    mse = torch.nn.MSELoss()
    # We will accumulate the mean loss in variable `loss`
    loss = torch.tensor(0., device=device)
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets, file_names = data
            inputs = inputs.to(device)
            # targets = targets.to(device)

            # Get outputs for network
            predictions = model(inputs)

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance

            # Calculate mean mse loss over all samples in dataloader (accumulate mean losses in `loss`)
            loss += (torch.stack([mse(output, torch.tensor(target.reshape((-1,))))
                                  for output, target in zip(predictions, targets)]).sum()
                     / len(dataloader.dataset))
    return loss


def padding_collate_fn(batch_as_list: list):
    # target_size = 100
    # get the minimum bounds in from list of input image shapes
    target_size_x = np.max([t[0][0].shape[0] for t in batch_as_list])
    target_size_y = np.max([t[0][0].shape[1] for t in batch_as_list])
    # for targets and ids, just use list of 2nd and 3rd element in tuple of each list element
    targets = [t[1] for t in batch_as_list]
    ids = [t[2] for t in batch_as_list]
    inputs = []
    for input_tensor, target_array, idx in batch_as_list:
        actual_size = input_tensor[0].shape
        pad_x_left = (target_size_x - actual_size[0]) // 2
        pad_x_right = target_size_x - actual_size[0] - pad_x_left
        pad_y_bottom = (target_size_y - actual_size[1]) // 2
        pad_y_top = target_size_y - actual_size[1] - pad_y_bottom
        new_input = torch.nn.functional.pad(input_tensor, pad=[pad_y_bottom, pad_y_top, pad_x_left, pad_x_right],
                                            mode='constant', value=0)
        inputs.append(new_input)

    inputs = torch.stack(inputs, dim=0)
    # targets = torch.stack(targets, dim=0)
    # ids = torch.stack(ids, dim=0)
    return inputs, targets, ids


def main(results_path, network_config: dict, eval_settings: dict, learning_rate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = int(1e5), device: torch.device = torch.device("cuda:0"), debug=False):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    # Prepare a path to plot to
    plotpath = os.path.join(results_path, 'plots')
    os.makedirs(plotpath, exist_ok=True)

    # Load or create the dataset
    greyscale_dataset = GreyscaleDataset(data_folder='data')

    # Split dataset into training, validation, and test set randomly
    trainingset = torch.utils.data.Subset(greyscale_dataset, indices=np.arange(int(len(greyscale_dataset) * (3 / 5))))
    validationset = torch.utils.data.Subset(greyscale_dataset, indices=np.arange(int(len(greyscale_dataset) * (3 / 5)),
                                                                                 int(len(greyscale_dataset) * (4 / 5))))
    testset = torch.utils.data.Subset(greyscale_dataset, indices=np.arange(int(len(greyscale_dataset) * (4 / 5)),
                                                                           len(greyscale_dataset)))

    # Create datasets and dataloaders with rotated targets without augmentation (for evaluation)
    trainingset_eval = CroppedImages(dataset=trainingset, debug=debug)
    validationset = CroppedImages(dataset=validationset, debug=debug)
    testset = CroppedImages(dataset=testset, debug=debug)
    trainloader = torch.utils.data.DataLoader(trainingset_eval, batch_size=1, shuffle=False, num_workers=0,
                                              collate_fn=padding_collate_fn)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=0,
                                            collate_fn=padding_collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0,
                                             collate_fn=padding_collate_fn)

    # Create datasets and dataloaders with rotated targets with augmentation (for training)
    chain = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    trainingset_augmented = CroppedImages(dataset=trainingset, transform_chain=chain)
    trainloader_augmented = torch.utils.data.DataLoader(trainingset_augmented, batch_size=16, shuffle=True,
                                                        num_workers=0, collate_fn=padding_collate_fn)

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))

    # Create Network
    net = DeCropCNN(**network_config)
    net.to(device)

    # Get mse loss function
    mse = torch.nn.MSELoss()

    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print_stats_at = eval_settings['print_stats_at']  # print status to tensorboard every x updates
    plot_at = eval_settings['plot_at']  # plot every x updates
    validate_at = eval_settings['validate_at']  # test on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))

    # Train until n_updates update have been reached
    while update < n_updates:
        for data in trainloader_augmented:
            # Get next samples in `trainloader_augmented`
            inputs, targets, ids = data
            inputs = inputs.to(device)
            # crop_arrays = inputs[:, 1, ...]
            # target_masks = crop_arrays.to(dtype=torch.bool)
            # Reset gradients
            optimizer.zero_grad()

            # Get outputs for network
            predictions = net(inputs)

            # Calculate loss, do backward pass, and update weights
            # loss = mse(outputs, targets)
            loss = torch.stack([mse(output, torch.tensor(target.reshape((-1,))))
                                for output, target in zip(predictions, targets)])
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            # Print current status and score
            if update % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)

            # Plot output
            if update % plot_at == 0:
                plot(inputs.detach().cpu().numpy(), targets, predictions, plotpath, update)

            # Evaluate model on validation set
            if update % validate_at == 0 and update > 0:
                val_loss = evaluate_model(net, dataloader=valloader, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(), global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}', values=param.grad.cpu(), global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    torch.save(net, os.path.join(results_path, 'best_model.pt'))

            update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progess_bar.update()

            # Increment update counter, exit if maximum number of updates is reached
            update += 1
            if update >= n_updates:
                break

    update_progess_bar.close()
    print('Finished Training!')

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    test_loss = evaluate_model(net, dataloader=testloader, device=device)
    val_loss = evaluate_model(net, dataloader=valloader, device=device)
    train_loss = evaluate_model(net, dataloader=trainloader, device=device)

    print(f"Scores:")
    print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")

    # Write result to file
    with open(os.path.join(results_path, 'results.txt'), 'w') as fh:
        print(f"Scores:", file=fh)
        print(f"test loss: {test_loss}", file=fh)
        print(f"validation loss: {val_loss}", file=fh)
        print(f"training loss: {train_loss}", file=fh)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as fh:
        config = json.load(fh)
    main(**config)

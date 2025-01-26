# Standard library imports
import os
import shutil

# Third-party imports
import torch
from torch import nn
import torch.nn.functional as F


def clear_directory_files(
    path: str, verbose: bool = False,
    create_if_missing: bool = True
) -> None:
    """
    Checks if the given directory has files and deletes all files within it.

    Parameters
    ----------
    path (str):
        The path to the directory where files will be checked and
        deleted.

    Returns:
        None
    """
    # Check if the directory exists
    if not os.path.isdir(path):
        if create_if_missing:
            os.makedirs(path)
            if verbose:
                print(f"Directory created: {path}")
        else:
            if verbose:
                print(f"The directory {path} does not exist.")
            return

    # List all entries in the directory
    for entry in os.listdir(path):
        # Construct full entry path
        entry_path = os.path.join(path, entry)
        # Check if it is a file and delete it
        if os.path.isfile(entry_path):
            os.remove(entry_path)
            if verbose:
                print(f"Deleted file: {entry_path}")
        elif os.path.isdir(entry_path):
            # If it's a directory and you want to remove directories as well
            # shutil.rmtree(entry_path)
            print(f"Skipped directory: {entry_path}")

    print(f"All files have been deleted from {path}.")


def next_multiple(number, multiple_of: int = 16):
    next_mul = ((number + (multiple_of - 1)) // multiple_of) * multiple_of
    return next_mul


def get_2d_padding_shape(
    tensor: torch.Tensor,
    multiple_of: int = 32
) -> torch.Tensor:

    pad_d0 = next_multiple(tensor.size(2), multiple_of) - tensor.size(2)
    pad_d1 = next_multiple(tensor.size(3), multiple_of) - tensor.size(3)

    padding_shape = [0, pad_d1, 0, pad_d0]

    return padding_shape


def strip_2d_padding(tensor, padding_shape):
    tensor = tensor[..., 0: tensor.size(2) - padding_shape[3], 0: tensor.size(3) - padding_shape[1]]
    return tensor


def predict_dimension(
    tensor: torch.Tensor,
    model: nn.Module,
    spatial_dim: int = 0
) -> torch.Tensor:
    """
    Predict on 3D data through a single dimension with a 2D model.

    This prediction function iterates through all the slices in a single
    axis/dimension, pads them to the nearest multiple of 16, puts them on cuda,
    and replaces the original data with the predictions.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (D, H, W)
    """

    assert tensor.dim() == 3, (
        f"Input tensor must have three dimensions; got {tensor.dim()}"
    )

    # Adapt to batch and channel mandate for input to model.
    dim = spatial_dim + 2
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    dim_size = tensor.size(dim)

    padding_shape = get_2d_padding_shape(tensor.select(dim, 0))

    with torch.no_grad():
        for index in range(dim_size):
            # Select the slice and convert to float and put on cuda.
            dim_slice = tensor.select(dim, index).cuda().float()
            # Pad slice
            dim_slice = F.pad(dim_slice, padding_shape, mode="reflect")
            # Make the prediction
            dim_slice_inference = model(dim_slice).to(tensor.dtype).to(
                tensor.device
            )
            # Remove the padding
            dim_slice_inference = strip_2d_padding(
                dim_slice_inference,
                padding_shape
            )
            # Override data with prediction
            tensor.select(dim, index).copy_(dim_slice_inference)

    return tensor.squeeze()

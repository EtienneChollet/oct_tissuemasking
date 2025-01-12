import os
import re
import sys
import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
import cornucopia as cc
import torch.nn.functional as F


class MultiConv3D(nn.Module):
    """
    A module that performs multiple consecutive convolutional operations, each
    followed by a specified normalization and a LeakyReLU activation function.
    It includes a residual connection that adds the input directly to the
    output of the convolutional blocks if their dimensions match.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    num_convs : int
        Number of convolutional layers in the sequence.
    normalization : str, optional
        Type of normalization to use. Options include 'batch', 'layer', or
        'instance'.
        Default is 'layer'.

    Attributes
    ----------
    convs : nn.Sequential
        A sequential container of convolutional blocks, each consisting of a
        convolution layer followed by normalization and LeakyReLU activation.
    skip : nn.Sequential or nn.Identity
        A skip connection that applies a 1x1 convolution if the number of input
        and output channels does not match, otherwise it is an identity layer.

    Methods
    -------
    _get_norm_layer(norm_type, out_channels)
        Determines the appropriate normalization layer based on the specified
        type.

    forward(x)
        Defines the computation performed at every call. Applies the 
        convolution blocks to the input tensor, adds the result to the output
        of the skip connection, and returns the final output.
    """

    def __init__(self, in_channels, out_channels, num_convs=2,
                 normalization='layer'):
        super(MultiConv3D, self).__init__()
        self.num_convs = num_convs
        norm_layer = self._get_norm_layer(normalization, out_channels)

        layers = []
        for i in range(num_convs):
            if i == 0:
                layers.append(nn.Conv3d(
                    in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv3d(
                    out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(norm_layer)
            layers.append(nn.LeakyReLU())

        self.convs = nn.Sequential(*layers)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                norm_layer
            )
        else:
            self.skip = nn.Identity()

    def _get_norm_layer(self, norm_type, out_channels):
        """
        Selects the appropriate normalization layer based on the provided type.

        Parameters
        ----------
        norm_type : str
            The type of normalization to use ('batch', 'layer', or 'instance').
        out_channels : int
            Number of output channels for which the normalization is applied.

        Returns
        -------
        nn.Module
            The normalization layer.

        Raises
        ------
        ValueError
            If the provided normalization type is not supported.
        """
        if norm_type == 'batch':
            return nn.BatchNorm3d(out_channels)
        elif norm_type == 'layer':
            return nn.GroupNorm(1, out_channels)
        elif norm_type == 'instance':
            return nn.GroupNorm(out_channels, out_channels)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def forward(self, x):
        """
        Applies the module's operations to the input tensor and returns the
        output.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to process.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the convolution blocks and the
            residual connection.
        """
        shortcut = self.skip(x)
        x = self.convs(x)
        x += shortcut
        return x


class UpConv3D(nn.Module):
    """
    Upscaling using transposed convolution followed by double conv with
    correct channel handling.
    """

    def __init__(self, in_channels, merge_channels, out_channels, num_convs=2,
                 normalization='layer'
                 ):
        super(UpConv3D, self).__init__()
        # Halve the channels for upscaling
        self.up = nn.ConvTranspose3d(
            in_channels, in_channels // 2,
            kernel_size=2,
            stride=2
            )
        # Double conv expects the concatenated channels
        self.conv = MultiConv3D(
            in_channels // 2 + merge_channels,
            out_channels,
            num_convs=num_convs,
            normalization=normalization)

    def forward(self, x, skip):
        x = self.up(x)
        # If there's a size mismatch, align before concatenation
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])

        # Concatenate along the channel dimension
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ScaledUNet(nn.Module):
    """
    Simple UNet model for 3D image segmentation with scalable filters.

    Parameters
    ----------
    n_channels : int
        Number of input channels.
    n_classes : int
        Number of output classes.
    base_filters : int, optional
        Number of base filters used in the convolutions. Default is 16.
    scale_factor : int, optional
        Factor by which the number of filters is scaled in each layer.
        Default is 2.
    dropout : float, optional
        Dropout rate. Default is 0.2.
    num_convs : int
        Number of convolutional layers in the sequence.
    normalization : str, optional
        Type of normalization to use. Options include 'batch', 'layer', or
        'instance'. Default is 'layer'.
    """
    def __init__(self, n_channels: int = 1, n_classes: int = 1,
                 base_filters: int = 16, scale_factor: int = 2,
                 dropout: float = 0.0, norm: str = "layer",
                 num_convs: int = 2):
        super(ScaledUNet, self).__init__()
        self.dropout = dropout
        self.scale_factor = scale_factor

        self.inc = MultiConv3D(
            n_channels,
            base_filters,
            normalization=norm,
            num_convs=num_convs
            )
        self.down1 = MultiConv3D(
            base_filters,
            base_filters * scale_factor,
            normalization=norm,
            num_convs=num_convs
            )
        self.down2 = MultiConv3D(
            base_filters * scale_factor,
            base_filters * scale_factor**2,
            normalization=norm,
            num_convs=num_convs
            )
        self.down3 = MultiConv3D(
            base_filters * scale_factor**2,
            base_filters * scale_factor**3,
            normalization=norm,
            num_convs=num_convs
            )
        self.down4 = MultiConv3D(
            base_filters * scale_factor**3,
            base_filters * scale_factor**4,
            normalization=norm,
            num_convs=num_convs
            )

        self.up1 = UpConv3D(
            base_filters * scale_factor**4, base_filters * scale_factor**3,
            base_filters * scale_factor**3)
        self.up2 = UpConv3D(
            base_filters * scale_factor**3, base_filters * scale_factor**2,
            base_filters * scale_factor**2)
        self.up3 = UpConv3D(
            base_filters * scale_factor**2, base_filters * scale_factor,
            base_filters * scale_factor)
        self.up4 = UpConv3D(
            base_filters * scale_factor, base_filters, base_filters)

        self.outc = nn.Conv3d(base_filters, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = nn.Dropout3d(self.dropout)(x1)

        x2 = F.max_pool3d(x1, 2)
        x2 = self.down1(x2)
        x2 = nn.Dropout3d(self.dropout)(x2)

        x3 = F.max_pool3d(x2, 2)
        x3 = self.down2(x3)
        x3 = nn.Dropout3d(self.dropout)(x3)

        x4 = F.max_pool3d(x3, 2)
        x4 = self.down3(x4)
        x4 = nn.Dropout3d(self.dropout)(x4)

        x5 = F.max_pool3d(x4, 2)
        x5 = self.down4(x5)
        x5 = nn.Dropout3d(self.dropout)(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


class ModelConfigManager:
    def __init__(self, base_model_dir, n_classes=3, verbose=False):
        """
        Initialize the manager with the directory and version of the model
        configuration.

        Parameters
        ----------
        base_model_dir : str
            Base directory of the model configurations.
        version : str
            Version of the configuration to use.
        """
        self.verbose = verbose
        self.n_classes = n_classes
        self.model_dir = base_model_dir

    def get_most_recent_checkpoint(self):
        """Get the most recent checkpoint from the directory.

        Returns
        -------
        str
            The file path of the most recent checkpoint.
        """
        checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
        checkpoint_paths = [
            os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
            if re.match(r'checkpoint_epoch_\d+_val-', f)]

        if not checkpoint_paths:
            return None

        # Sort the checkpoints by extracting the epoch number and sort in
        # descending order
        checkpoint_paths.sort(
            key=lambda x: int(
                re.search(r'epoch_(\d+)_', x).group(1)), reverse=True)
        checkpoint_path = checkpoint_paths[0]
        # if self.verbose is True:
        print(checkpoint_path)
        return checkpoint_path

    def build_and_load_model(self, channels=3, device='cuda'):
        """Build the model from configuration and load the latest checkpoint.

        Parameters
        ----------
        device : str, optional
            The device to load the model onto ('cuda' or 'cpu').

        Returns
        -------
        object
            The loaded PyTorch model.
        """
        # Make sure that models trained with dropout and batch norm are not
        # predicting with it

        model = ScaledUNet(base_filters=8, scale_factor=1)

        checkpoint_path = self.get_most_recent_checkpoint()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
        return model


class UNetPredictor:
    def __init__(self, model, device='cuda'):
        """
        Initialize the predictor with the model checkpoint and device.

        Parameters
        ----------
        model_path : str
            Path to the saved model checkpoint.
        device : torch.device, optional
            Device to run the model on. If None, uses CUDA if available.
        """
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = model.to('cuda')

    def predict(self, image_path):
        """
        Make a prediction using an image file path.

        Parameters
        ----------
        image_path : str
            Path to the image to predict.

        Returns
        -------
        torch.Tensor
            The model's prediction as a tensor.
        """
        image = self.preprocess_input(image_path)
        return self.predict_tensor(image)

    def preprocess_input(self, image_path):
        """
        Load and preprocess an input image from a file path.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        torch.Tensor
            Preprocessed image tensor.
        """
        image = nib.load(image_path).get_fdata()
        return image

    def predict_tensor(self, tensor):
        """
        Make a prediction using a pre-loaded tensor. Ensures the tensor has
        exactly four dimensions (e.g., [batch, channels, height, width]).

        Parameters
        ----------
        tensor : torch.Tensor
            A pre-loaded and preprocessed tensor ready for prediction.
            Expected to have four dimensions.

        Returns
        -------
        torch.Tensor
            The model's prediction as a tensor.

        Raises
        ------
        ValueError
            If the tensor does not have exactly four dimensions.
        """
        if tensor.dim() < 4:
            tensor = tensor.unsqueeze(0)
        # tensor = QuantileTransform(vmin=-0.5, vmax=0.5)(tensor)
        with torch.no_grad():
            self.model = self.model.eval()
            prediction = self.model(tensor.to('cuda'))
        return prediction


class FullPredict:

    def __init__(self, tensor, model, patch_size=128, step_size=64,
                 padding='reflect'):
        self.tensor = tensor
        # self.norm = transforms.Normalize(tensor.mean(), tensor.std())
        self.patch_size = patch_size
        self.step_size = step_size
        self.padding = padding
        self._padit()
        self.model = model
        self.imprint_tensor = torch.zeros((1, 1, self.tensor.shape[0],
                                           self.tensor.shape[1],
                                           self.tensor.shape[2]),
                                          dtype=torch.float32,
                                          device=tensor.device)
        self.complete_patch_coords = self._get_patch_coords()
        self.num_patches = len(self.complete_patch_coords)

    def predict(self, gaussian_sigma=10):
        self.model.eval()
        total_patches = len(self.complete_patch_coords)
        print(f"Starting prediction on {total_patches} patches...")

        with torch.no_grad():
            for idx, i in enumerate(self.complete_patch_coords):
                in_tensor = self.tensor[tuple(i)].unsqueeze(0).unsqueeze(0).float()
                try:
                    in_tensor = cc.QuantileTransform(
                        vmin=-1, vmax=1
                        )(in_tensor)
                except:
                    pass
                # Predict (w/ automated mixed precision) and add to imprint
                prediction = self.model(in_tensor).float().sigmoid()
                prediction = 1 - prediction
                self.imprint_tensor[..., i[0], i[1], i[2]] += prediction.float()
                del prediction, in_tensor
                torch.cuda.empty_cache()

                # Progress output
                if idx % 10 == 0 or idx == total_patches - 1:  # Update every 10 patches or on the last patch
                    sys.stdout.write(
                        f"\rPredicting!: Patch {idx + 1}/{total_patches} ({(idx + 1) / total_patches * 100:.2f}%)")
                    sys.stdout.flush()

            self._reformat_imprint_tensor()

    def _get_patch_coords(self):
        patch_size = self.patch_size
        step_size = self.step_size
        parent_shape = self.tensor.shape
        complete_patch_coords = []
        x_coords = [slice(x, x + patch_size) for x in range(
            step_size, parent_shape[0] - patch_size, step_size)]
        y_coords = [slice(y, y + patch_size) for y in range(
            step_size, parent_shape[1] - patch_size, step_size)]
        z_coords = [slice(z, z + patch_size) for z in range(
            step_size, parent_shape[2] - patch_size, step_size)]
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    complete_patch_coords.append((x, y, z))
        complete_patch_coords = np.array(complete_patch_coords)
        return complete_patch_coords

    def _reformat_imprint_tensor(self):
        if self._got_padded:
            self.imprint_tensor = self.imprint_tensor[
                :, :,
                self.patch_size:-self.patch_size,
                self.patch_size:-self.patch_size,
                self.patch_size:-self.patch_size
                ]
        redundancy = (self.patch_size ** 3) // (self.step_size ** 3)
        print(f"\n\n{redundancy}x Averaging...")
        self.imprint_tensor /= redundancy

    def _padit(self):
        pad = [self.patch_size] * 6
        self.tensor = torch.nn.functional.pad(self.tensor.unsqueeze(0), pad,
                                              mode=self.padding)
        self.tensor = self.tensor.squeeze()
        self._got_padded = True

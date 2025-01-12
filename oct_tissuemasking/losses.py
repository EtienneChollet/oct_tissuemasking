"""
Module for loss functions

Notes
-----
MOST OF THESE LOSSES WERE TAKEN FROM https://github.com/balbasty
"""

from torch import nn
import torch
import inspect
import torch.nn.functional as F


def make_vector(input, n=None, crop=True, *args,
                dtype=None, device=None, **kwargs):
    """Ensure that the input is a (tensor) vector and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    dtype : torch.dtype, optional
        Output data type.
    device : torch.device, optional
        Output device

    Returns
    -------
    output : tensor
        Output vector.

    """
    input = torch.as_tensor(input, dtype=dtype, device=device).flatten()
    if n is None:
        return input
    if n is not None and input.numel() >= n:
        return input[:n] if crop else input
    if args:
        default = args[0]
    elif 'default' in kwargs:
        default = kwargs['default']
    else:
        default = input[-1]
    default = input.new_full([n-len(input)], default)
    return torch.cat([input, default])


def _dot(x, y):
    """Dot product along the last dimension"""
    return x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1).squeeze(-1)


def _make_activation(activation):
    if isinstance(activation, str):
        activation = getattr(nn, activation)
    activation = (activation() if inspect.isclass(activation)
                  else activation if callable(activation)
                  else None)
    return activation


class Loss(nn.Module):
    """Base class for losses"""

    def __init__(self, reduction='mean'):
        """
        Parameters
        ----------
        reduction : {'mean', 'sum'} or callable
            Reduction to apply across batch elements
        """
        super().__init__()
        self.reduction = reduction

    def reduce(self, x):
        if not self.reduction:
            return x
        if isinstance(self.reduction, str):
            if self.reduction.lower() == 'mean':
                return x.mean()
            if self.reduction.lower() == 'sum':
                return x.sum()
            raise ValueError(f'Unknown reduction "{self.reduction}"')
        if callable(self.reduction):
            return self.reduction(x)
        raise ValueError(f'Don\'t know what to do with reduction: '
                         f'{self.reduction}')


class DiceLoss(Loss):
    r"""Soft Dice Loss

    By default, each class is weighted identically.
    The `weighted` mode allows classes to be weighted by frequency.

    References
    ----------
    ..  "V-Net: Fully convolutional neural networks for volumetric
         medical image segmentation"
        Milletari, Navab and Ahmadi
        3DV (2016)
        https://arxiv.org/abs/1606.04797
    ..  "Generalised dice overlap as a deep learning loss function for
         highly unbalanced segmentations"
        Sudre, Li, Vercauteren, Ourselin and Cardoso
        DLMIA (2017)
        https://arxiv.org/abs/1707.03237
    ..  "The Dice loss in the context of missing or empty labels:
         introducing $\Phi$ and $\epsilon$"
        Tilborghs, Bertels, Robben, Vandermeulen and Maes
        MICCAI (2022)
        https://arxiv.org/abs/2207.09521
    """

    def __init__(self, square=True, weighted=False, labels=None,
                 eps=None, reduction='mean', activation=None):
        """

        Parameters
        ----------
        square : bool, default=True
            Square the denominator in SoftDice.
        weighted : bool or list[float], default=False
            If True, weight the Dice of each class by its frequency in the
            reference. If a list, use these weights for each class.
        labels : list[int], default=range(nb_class)
            Label corresponding to each one-hot class. Only used if the
            reference is an integer label map.
        eps : float or list[float], default=1/K
            Stabilization of the Dice loss.
            Optimally, should be equal to each class' expected frequency
            across the whole dataset. See Tilborghs et al.
        reduction : {'mean', 'sum', None} or callable, default='mean'
            Type of reduction to apply across minibatch elements.
        activation : nn.Module or str
            Activation to apply to the prediction before computing the loss
        """
        super().__init__(reduction)
        self.square = square
        self.weighted = weighted
        self.labels = labels
        self.eps = eps
        self.activation = _make_activation(activation)

    def forward_onehot(self, pred, ref, mask, weights, eps):

        nb_classes = pred.shape[1]
        if ref.shape[1] != nb_classes:
            raise ValueError(f'Number of classes not consistent. '
                             f'Expected {nb_classes} but got {ref.shape[1]}.')

        ref = ref.to(pred)
        if mask is not None:
            pred = pred * mask
            ref = ref * mask
        pred = pred.reshape([*pred.shape[:2], -1])       # [B, C, N]
        ref = ref.reshape([*ref.shape[:2], -1])          # [B, C, N]

        # Compute SoftDice
        inter = _dot(pred, ref)                          # [B, C]
        if self.square:
            pred = pred.square()
            ref = ref.square()
        pred = pred.sum(-1)                              # [B, C]
        ref = ref.sum(-1)                                # [B, C]
        union = pred + ref
        # loss = (2 * inter + eps) / (union + eps)
        loss = (2 * inter + eps) / (union + eps)

        # Simple or weighted average
        if weights is not False:
            if weights is True:
                # weights = ref // ref.sum(dim=1, keepdim=True)
                weights = ref / ref.sum(dim=1, keepdim=True)
            loss = loss * weights
            loss = loss.sum(-1)
        else:
            loss = loss.mean(-1)

        # Minibatch reduction
        loss = 1 - loss
        return self.reduce(loss)

    def forward_labels(self, pred, ref, mask, weights, eps):

        nb_classes = pred.shape[1]
        labels = self.labels or list(range(nb_classes))

        loss = 0
        sumweights = 0
        for index, label in enumerate(labels):
            if label is None:
                continue
            pred1 = pred[:, index]
            eps1 = eps[index]
            ref1 = (ref == label).squeeze(1)
            if mask is not None:
                pred1 = pred1 * mask
                ref1 = ref1 * mask

            pred1 = pred1.reshape([len(pred1), -1])           # [B, N]
            ref1 = ref1.reshape([len(ref1), -1])              # [B, N]

            # Compute SoftDice
            inter = (pred1 * ref1).sum(-1)                    # [B]
            if self.square:
                pred1 = pred1.square()
            pred1 = pred1.sum(-1)                             # [B]
            ref1 = ref1.sum(-1)                               # [B]
            union = pred1 + ref1
            loss1 = (2 * inter + eps1) / (union + eps1)
            # print(f'Loss1: {loss1}')

            # Simple or weighted average
            if weights is not False:
                if weights is True:
                    weight1 = ref1
                else:
                    weight1 = float(weights[index])
                loss1 = loss1 * weight1
                sumweights += weight1
            else:
                sumweights += 1
            loss += loss1

        # Minibatch reduction
        loss = loss / sumweights
        loss = 1 - loss
        return self.reduce(loss)

    def forward(self, pred, ref, mask=None):
        """

        Parameters
        ----------
        pred : (batch, nb_class, *spatial) tensor
            Predicted classes.
        ref : (batch, nb_class|1, *spatial) tensor
            Reference classes (or their expectation).
        mask : (batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or (batch,) tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar tensor.

        """
        if self.activation:
            pred = self.activation(pred)

        nb_classes = pred.shape[1]
        backend = dict(dtype=pred.dtype, device=pred.device)
        nvox = pred.shape[2:].numel()

        eps = self.eps or 1/nb_classes
        eps = make_vector(eps, nb_classes, **backend)
        eps = eps * nvox

        # prepare weights
        weighted = self.weighted
        if not torch.is_tensor(weighted) and not weighted:
            weighted = False
        if not isinstance(weighted, bool):
            weighted = make_vector(weighted, nb_classes, **backend)

        if ref.dtype.is_floating_point:
            return self.forward_onehot(pred, ref, mask, weighted, eps)
        else:
            return self.forward_labels(pred, ref, mask, weighted, eps)


class CatLoss(Loss):
    r"""Weighted categorical cross-entropy.

    By default, each class is weighted *identically*.
    /!\ This differs from the classical "categorical cross-entropy loss",
    /!\ which corresponds to the true Categorical log-likelihood and where
    /!\ classes are therefore weighted by frequency. The default behavior
    /!\ of our loss is that of a "weighted categorical cross-entropy".
    The `weighted` mode allows classes to be weighted by frequency.
    """

    def __init__(self, weighted=False, labels=None,
                 reduction='mean', activation=None):
        """

        Parameters
        ----------
        weighted : bool or list[float], default=False
            If True, weight the term of each class by its frequency
             in the reference. If a list, use these weights for each class.
        labels : list[int], default=range(nb_class)
            Label corresponding to each one-hot class. Only used if the
            reference is an integer label map.
        reduction : {'mean', 'sum', None} or callable, default='mean'
            Type of reduction to apply across minibatch elements.
        activation : nn.Module or str
            Activation to apply to the prediction before computing the loss
        """
        super().__init__(reduction)
        self.weighted = weighted
        self.labels = labels
        self.reduction = reduction
        self.activation = _make_activation(activation)

    def forward_onehot(self, pred, ref, mask, weights):

        nb_classes = pred.shape[1]
        if ref.shape[1] != nb_classes:
            raise ValueError(f'Number of classes not consistent. '
                             f'Expected {nb_classes} but got {ref.shape[1]}.')

        ref = ref.to(pred)
        if mask is not None:
            pred = pred * mask
            ref = ref * mask

        # Compute dot(ref, log(pred)) / dot(ref, 1)
        pred = pred.reshape([*pred.shape[:2], -1])       # [B, C, N]
        ref = ref.reshape([*ref.shape[:2], -1])          # [B, C, N]
        loss = _dot(pred, ref)                           # [B, C]
        ref = ref.sum(-1)                                # [B, C]
        loss = loss / ref                                # [B, C]

        # Simple or weighted average
        if weights is not False:
            if weights is True:
                weights = ref / ref.sum(dim=1, keepdim=True)
            loss = loss * weights
            loss = loss.sum(-1)
        else:
            loss = loss.mean(-1)

        # Minibatch reduction
        return self.reduce(loss.neg_())

    def forward_labels(self, pred, ref, mask, weights):

        nb_classes = pred.shape[1]
        labels = self.labels or list(range(nb_classes))

        loss = 0
        sumweights = 0
        for index, label in enumerate(labels):
            if label is None:
                continue
            pred1 = pred[:, index]
            ref1 = (ref == label).squeeze(1)
            if mask is not None:
                pred1 = pred1 * mask
                ref1 = ref1 * mask

            pred1 = pred1.reshape([len(pred1), -1])           # [B, N]
            ref1 = ref1.reshape([len(ref1), -1])              # [B, N]

            # Compute SoftDice
            loss1 = (pred1 * ref1).sum(-1)                    # [B]
            ref1 = ref1.sum(-1)                               # [B]
            loss1 = loss1 / ref1.clamp_min_(1e-5)

            # Simple or weighted average
            if weights is not False:
                if weights is True:
                    weight1 = ref1
                else:
                    weight1 = float(weights[index])
                loss1 = loss1 * weight1
                sumweights += weight1
            else:
                sumweights += 1
            loss += loss1

        # Minibatch reduction
        loss = loss / sumweights
        loss = 1 - loss
        return self.reduce(loss)

    def forward(self, pred, ref, mask=None):
        """

        Parameters
        ----------
        pred : (batch, nb_class, *spatial) tensor
            Predicted classes.
        ref : (batch, nb_class|1, *spatial) tensor
            Reference classes (or their expectation).
        mask : (batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or (batch,) tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar tensor.

        """
        if self.activation:
            pred = self.activation(pred)

        nb_classes = pred.shape[1]
        backend = dict(dtype=pred.dtype, device=pred.device)

        pred = pred.log()
        pred.masked_fill_(~torch.isfinite(pred), 0)

        # prepare weights
        weighted = self.weighted
        if not torch.is_tensor(weighted) and not weighted:
            weighted = False
        if not isinstance(weighted, bool):
            weighted = make_vector(weighted, nb_classes, **backend)

        if ref.dtype.is_floating_point:
            return self.forward_onehot(pred, ref, mask, weighted)
        else:
            return self.forward_labels(pred, ref, mask, weighted)


class CatMSELoss(Loss):
    """Mean Squared Error between one-hots."""

    def __init__(self, weighted=False, labels=None, reduction='mean',
                 activation=None):
        """

        Parameters
        ----------
        weighted : bool or list[float], default=False
            If True, weight the Dice of each class by its size in the
            reference. If a list, use these weights for each class.
        labels : list[int], default=range(nb_class)
            Label corresponding to each one-hot class. Only used if the
            reference is an integer label map.
        reduction : {'mean', 'sum', None} or callable, default='mean'
            Type of reduction to apply across minibatch elements.
        activation : nn.Module or str
            Activation to apply to the prediction before computing the loss
        """
        super().__init__(reduction)
        self.weighted = weighted
        self.labels = labels
        self.reduction = reduction
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        self.activation = activation

    def forward_onehot(self, pred, ref, mask, weights):

        nb_classes = pred.shape[1]
        if ref.shape[1] != nb_classes:
            raise ValueError(f'Number of classes not consistent. '
                             f'Expected {nb_classes} but got {ref.shape[1]}.')

        ref = ref.to(pred)
        if mask is not None:
            pred = pred * mask
            ref = ref * mask
            mask = mask.reshape([*mask.shape[:2], -1])

        pred = pred.reshape([*pred.shape[:2], -1])       # [B, C, N]
        ref = ref.reshape([*ref.shape[:2], -1])          # [B, C, N]
        loss = pred - ref
        loss = _dot(loss, loss)                          # [B, C]
        loss = loss / (mask.sum(-1) if mask is not None else pred.shape[-1])

        # Simple or weighted average
        if weights is not False:
            if weights is True:
                weights = ref / ref.sum(dim=1, keepdim=True)
            loss = loss * weights
            loss = loss.sum(-1)
        else:
            loss = loss.mean(-1)

        # Minibatch reduction
        return self.reduce(loss)

    def forward_labels(self, pred, ref, mask, weights):

        nb_classes = pred.shape[1]
        labels = self.labels or list(range(nb_classes))

        loss = 0
        sumweights = 0
        for index, label in enumerate(labels):
            if label is None:
                continue
            pred1 = pred[:, index]
            ref1 = (ref == label).squeeze(1)
            if mask is not None:
                pred1 = pred1 * mask
                ref1 = ref1 * mask
                mask1 = mask.reshape([len(mask), -1])

            pred1 = pred1.reshape([len(pred1), -1])           # [B, N]
            ref1 = ref1.reshape([len(ref1), -1])              # [B, N]

            # Compute SoftDice
            loss1 = pred1 - ref1
            loss1 = _dot(loss1, loss1)
            loss1 = loss1 / (mask1.sum(-1) if mask is not None
                             else pred1.shape[-1])

            # Simple or weighted average
            if weights is not False:
                if weights is True:
                    weight1 = ref1
                else:
                    weight1 = float(weights[index])
                loss1 = loss1 * weight1
                sumweights += weight1
            else:
                sumweights += 1
            loss += loss1

        # Minibatch reduction
        loss = loss / sumweights
        return self.reduce(loss)

    def forward(self, pred, ref, mask=None):
        """

        Parameters
        ----------
        pred : (batch, nb_class, *spatial) tensor
            Predicted classes.
        ref : (batch, nb_class|1, *spatial) tensor
            Reference classes (or their expectation).
        mask : (batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or (batch,) tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar tensor.

        """
        if self.activation:
            pred = self.activation(pred)

        nb_classes = pred.shape[1]
        backend = dict(dtype=pred.dtype, device=pred.device)

        # prepare weights
        weighted = self.weighted
        if not torch.is_tensor(weighted) and not weighted:
            weighted = False
        if not isinstance(weighted, bool):
            weighted = make_vector(weighted, nb_classes, **backend)

        if ref.dtype.is_floating_point:
            return self.forward_onehot(pred, ref, mask, weighted)
        else:
            return self.forward_labels(pred, ref, mask, weighted)


class LogitMSELoss(Loss):
    """
    Mean Squared Error between logits and target positive/negative values."""

    def __init__(self, target=5, weighted=False, labels=None, reduction='mean',
                 activation=None):
        """

        Parameters
        ----------
        target : float
            Target value when the ground truth is True.
        weighted : bool or list[float] or 'inv', default=False
            If True, weight the score of each class by its frequency in
            the reference.
            If 'inv', weight the score of each class by its inverse
            frequency in the reference.
            If a list, use these weights for each class.
        labels : list[int], default=range(nb_class)
            Label corresponding to each one-hot class. Only used if the
            reference is an integer label map.
        reduction : {'mean', 'sum', None} or callable, default='mean'
            Type of reduction to apply across minibatch elements.
        activation : nn.Module or str
            Activation to apply to the prediction before computing the loss
        """
        super().__init__(reduction)
        self.weighted = weighted
        self.labels = labels
        self.reduction = reduction
        self.target = target
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        self.activation = activation

    def forward_onehot(self, pred, ref, mask, weights):

        nb_classes = pred.shape[1]
        if ref.shape[1] != nb_classes:
            raise ValueError(f'Number of classes not consistent. '
                             f'Expected {nb_classes} but got {ref.shape[1]}.')

        ref = ref.to(pred)
        if mask is not None:
            pred = pred * mask
            ref = ref * mask
            mask = mask.reshape([*mask.shape[:2], -1])

        pred = pred.reshape([*pred.shape[:2], -1])       # [B, C, N]
        ref = ref.reshape([*ref.shape[:2], -1])          # [B, C, N]
        loss = pred + (1 - 2 * ref) * self.target
        loss = _dot(loss, loss)                          # [B, C]
        loss = loss / (mask.sum(-1) if mask is not None else pred.shape[-1])

        # Simple or weighted average
        if weights is not False:
            if weights is True:
                weights = ref.sum(dim=-1)
                weights = weights / weights.sum(dim=-1, keepdim=True)
            elif isinstance(weights, str) and weights[0].lower() == 'i':
                weights = ref.sum(dim=-1)
                weights = ref.shape[-1] - weights
                weights = weights / weights.sum(dim=-1, keepdim=True)
            loss = (loss * weights).sum(-1)
        else:
            loss = loss.mean(-1)

        # Minibatch reduction
        return self.reduce(loss)

    def forward_labels(self, pred, ref, mask, weights):

        nb_classes = pred.shape[1]
        labels = self.labels or list(range(nb_classes))

        loss = 0
        sumweights = 0
        for index, label in enumerate(labels):
            if label is None:
                continue
            pred1 = pred[:, index]
            ref1 = (ref == label).squeeze(1)
            if mask is not None:
                pred1 = pred1 * mask
                ref1 = ref1 * mask
                mask1 = mask.reshape([len(mask), -1])

            pred1 = pred1.reshape([len(pred1), -1])           # [B, N]
            ref1 = ref1.reshape([len(ref1), -1])              # [B, N]

            # Compute SoftDice
            loss1 = pred1 + (1 - 2 * ref1) * self.target
            loss1 = _dot(loss1, loss1)
            loss1 = loss1 / (mask1.sum(-1) if mask is not None
                             else pred1.shape[-1])

            # Simple or weighted average
            if weights is not False:
                if weights is True:
                    weight1 = ref1.sum(-1)
                elif isinstance(weights, str) and weights[0].lower() == 'i':
                    weight1 = ref1.shape[-1] - ref1.sum(-1)
                else:
                    weight1 = float(weights[index])
                loss1 = loss1 * weight1
                sumweights += weight1
            else:
                sumweights += 1
            loss += loss1

        # Minibatch reduction
        loss = loss / sumweights
        return self.reduce(loss)

    def forward(self, pred, ref, mask=None):
        """

        Parameters
        ----------
        pred : (batch, nb_class, *spatial) tensor
            Predicted classes.
        ref : (batch, nb_class|1, *spatial) tensor
            Reference classes (or their expectation).
        mask : (batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or (batch,) tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar tensor.

        """
        if self.activation:
            pred = self.activation(pred)

        nb_classes = pred.shape[1]
        backend = dict(dtype=pred.dtype, device=pred.device)

        # prepare weights
        weighted = self.weighted
        if not torch.is_tensor(weighted) and not weighted:
            weighted = False
        if not isinstance(weighted, bool):
            weighted = make_vector(weighted, nb_classes, **backend)

        if ref.dtype.is_floating_point:
            return self.forward_onehot(pred, ref, mask, weighted)
        else:
            return self.forward_labels(pred, ref, mask, weighted)


class FocalTverskyLoss3D(nn.Module):
    """
    Implementation of Focal Tversky Loss for multi-class 3D segmentation.

    Parameters
    ----------
    alpha : float
        Weight for false positives.
    beta : float
        Weight for false negatives.
    gamma : float
        Focusing parameter to adjust the penalization of hard examples.
    smooth : float
        Small value to avoid division by zero.

    Attributes
    ----------
    alpha : float
        See parameters.
    beta : float
        See parameters.
    gamma : float
        See parameters.
    smooth : float
        See parameters.

    Examples
    --------
    >>> loss = FocalTverskyLoss3D(alpha=0.7, beta=0.3, gamma=1.33)
    >>> inputs = torch.randn(10, 3, 64, 64, 64, requires_grad=True)
    >>> targets = torch.empty(10, 3, 64, 64, 64).random_(2)
    >>> output = loss(inputs, targets)
    >>> output.backward()
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-4):
        super(FocalTverskyLoss3D, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Forward pass to compute Focal Tversky Loss for 3D multi-class data.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted logits or probabilities.
        targets : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Computed Focal Tversky Loss.
        """
        # Ensure inputs are in probability space
        inputs = torch.softmax(inputs, dim=1)
        # Clamp to avoid exact 0 or 1
        inputs = inputs.clamp(min=self.smooth, max=1.0 - self.smooth)
        # Ensure targets are binary
        targets = targets.float()

        # True positives, false positives, and false negatives
        tp = torch.sum(inputs * targets, dim=(0, 2, 3, 4))
        fp = torch.sum(inputs * (1 - targets), dim=(0, 2, 3, 4))
        fn = torch.sum((1 - inputs) * targets, dim=(0, 2, 3, 4))

        # Tversky index for each class
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
            )

        # Focal Tversky loss for each class
        focal_tversky_loss = torch.pow((1 - tversky_index), self.gamma)

        # Average over all classes
        return torch.mean(focal_tversky_loss)


class WeightedFocalTverskyLoss(nn.Module):
    """
    Weighted Focal Tversky Loss for multi-class image segmentation.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of weights for each class.
    smooth : float, optional
        Smoothing factor to avoid division by zero. Default is 1e-6.
    alpha : float, optional
        Weight for false positives. Default is 0.7.
    beta : float, optional
        Weight for false negatives. Default is 0.3.
    gamma : float, optional
        Focusing parameter for focal loss. Default is 0.75.
    """
    def __init__(self, weights: torch.Tensor = [1, 1, 1], smooth: float = 1e-6,
                 alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75):
        super(WeightedFocalTverskyLoss, self).__init__()
        self.weights = weights  # Tensor of weights for each class
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
        """
        Compute the Weighted Focal Tversky Loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted logits of shape (N, C, D, H, W), where N is the batch
            size, C is the number of classes, and D, H, W are the depth,
            height, and width dimensions.
        targets : torch.Tensor
            Ground truth one-hot encoded labels of shape (N, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        # Ensure weights are on the same device as inputs
        # weights = self.weights.to(inputs.device)
        a = 1 / torch.count_nonzero(targets[:, 0, ...]).item()
        b = 1 / torch.count_nonzero(targets[:, 1, ...]).item()
        c = 1 / torch.count_nonzero(targets[:, 2, ...]).item()

        tot = a + b + c

        a = a / tot  # (tot - a) / tot
        b = b / tot  # (tot - b) / tot
        c = c / tot  # (tot - c) / tot

        weights = torch.tensor([a, b, c]).to(inputs.device)
        # print(weights.sum())

        inputs_soft = F.softmax(inputs, dim=1)

        # True Positives, False Positives & False Negatives
        TP = (inputs_soft * targets).sum(dim=(2, 3, 4))
        FP = ((1 - targets) * inputs_soft).sum(dim=(2, 3, 4))
        FN = (targets * (1 - inputs_soft)).sum(dim=(2, 3, 4))

        # Weighted Tversky Index
        weighted_tversky = (
            weights * ((TP + self.smooth) /
                       (TP + self.alpha * FP + self.beta * FN + self.smooth))
        ).sum()

        # Applying the focal component
        focal_tversky = (1 - weighted_tversky) ** self.gamma

        return focal_tversky


class BinaryDiceLoss(nn.Module):
    """
    Implementation of Dice Loss for binary segmentation tasks.
    """
    def __init__(self, smooth=1e-6):
        """
        Parameters:
        - smooth (float): A small constant to avoid division by zero.
        """
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        """
        Computes the Dice Loss for binary data.
        Assumes that `logits` are not yet passed through a sigmoid activation
        function.

        Parameters:
        - logits (torch.Tensor): The raw output of the network, with shape
        (N, 1, H, W) or (N, H, W)
        - true (torch.Tensor): The binary ground truth labels, with the same
        shape as `logits`

        Returns:
        - torch.Tensor: The scalar Dice Loss.
        """
        # Apply sigmoid to logits to get the probability map
        # probs = torch.sigmoid(logits)
        probs = logits

        # Flatten the tensors to simplify the computation
        probs_flat = probs.view(-1)
        true_flat = true.view(-1)

        # Calculate the intersection and union
        intersection = torch.sum(probs_flat * true_flat)
        union = torch.sum(probs_flat) + torch.sum(true_flat)

        # Compute the Dice score and then the loss
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        return dice_loss

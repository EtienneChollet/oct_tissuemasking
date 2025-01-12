import torch
# from .data import get_loaders
from oct_tissuemasking import losses
from oct_tissuemasking.utils import clear_directory_files
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save checkpoint to disk."""
    torch.save(state, filename)


def log_model_graph(model_dir: str, model: torch.nn.Module,
                    train_loader: DataLoader) -> SummaryWriter:
    """
    Logs the model graph to TensorBoard.

    Parameters
    ----------
    model_dir : str
        Directory where the TensorBoard logs will be saved.
    model : torch.nn.Module
        The model to be logged.
    train_loader : DataLoader
        DataLoader for the training data to provide a sample input.

    Returns
    -------
    SummaryWriter
        TensorBoard SummaryWriter object.
    """
    writer = SummaryWriter(model_dir)
    sample_inputs, _ = next(iter(train_loader))
    writer.add_graph(model, sample_inputs.to(next(model.parameters()).device))
    return writer


def log_metrics(writer: SummaryWriter, phase: str, metrics: dict, step: int
                ) -> None:
    """
    Logs training and validation metrics to TensorBoard.

    Parameters
    ----------
    writer : SummaryWriter
        TensorBoard writer object.
    phase : str
        Phase of training (e.g., 'training', 'validation', 'epoch').
    metrics : dict
        Dictionary of metrics to log.
    step : int
        Step (iteration or epoch) at which metrics are logged.
    """
    for key, value in metrics.items():
        writer.add_scalar(f'{phase}_{key}', value, step)


def train_one_epoch(model: torch.nn.Module, epoch: int, writer: SummaryWriter,
                    train_loader: DataLoader, criterion: torch.nn.Module,
                    optimizer: torch.optim.Adam, device: torch.device
                    ) -> float:
    """
    Trains the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    epoch : int
        Current epoch number.
    writer : SummaryWriter
        TensorBoard writer object for logging.
    train_loader : DataLoader
        DataLoader for the training data.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim
        Optimizer.
    device : torch.device
        Device to run the training on (e.g., 'cuda' or 'cpu').
    """
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        # labels = labels.float()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        print(f"E-{epoch+1}, I-{i+1}, Loss: {loss.item()}", end='\r', flush=True)

        # if i % 10 == 0:
        log_metrics(writer, 'training', {
            'loss': loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr']},
                    epoch * len(train_loader) + i)

    epoch_loss = float(running_loss / len(train_loader.dataset))
    log_metrics(writer, 'epoch', {'loss': epoch_loss}, epoch)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')


def validate(model: torch.nn.Module, epoch: int, writer: SummaryWriter,
             val_loader: DataLoader, optimizer: torch.optim.Adam,
             criterion: torch.nn.Module, device: torch.device,
             best_vloss: float, model_dir: str) -> float:
    """
    Validates the model on the validation set.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be validated.
    epoch : int
        Current epoch number.
    writer : SummaryWriter
        TensorBoard writer object for logging.
    val_loader : DataLoader
        DataLoader for the validation data.
    optimizer : Adam
        Optimizer.
    criterion : torch.nn.Module
        Loss function.
    device : torch.device
        Device to run the validation on (e.g., 'cuda' or 'cpu').
    best_vloss : float
        Best validation loss observed so far.
    model_dir : str
        Directory to save the model checkpoints.

    Returns
    -------
    float
        Best validation loss observed so far.
    """
    model.eval()
    running_vloss = 0.0

    with torch.no_grad():
        for vinputs, vlabels in val_loader:
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss.item()
    val_loss = float(running_vloss / len(val_loader))
    writer.add_scalar('val_loss', val_loss, epoch)

    if val_loss < best_vloss:
        best_vloss = val_loss
        print(f"New best val_loss: {best_vloss}")
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=(
            f'{model_dir}/checkpoints/'
            f'checkpoint_epoch_{epoch+1}_val-{val_loss}.pth.tar')
            )
    return best_vloss


def log_hist(model: torch.nn.Module, epoch: int, writer: SummaryWriter
             ) -> None:
    """
    Logs histograms of model parameters and their gradients to TensorBoard.

    Parameters
    ----------
    model : Module
        The model whose parameters are to be logged.
    epoch : int
        The current epoch number.
    writer : SummaryWriter
        TensorBoard writer object.
    """
    if (epoch + 1) % 10 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'{name}.grad', param.grad, epoch)


def configure_criterion(_type: str = 'dice', weighted: bool = False
                        ) -> torch.nn.Module:
    """
    Configures and returns the DiceLoss criterion (loss function).

    Parameters
    ----------
    weighted : bool, optional
        If True, use a weighted version of DiceLoss. Default is False.

    Returns
    -------
    torch.nn.Module
        Configured DiceLoss criterion.
    """
    if _type == 'dice':
        criterion_ = losses.DiceLoss(
            weighted=weighted, activation='Sigmoid')
    elif _type == 'focal':
        criterion_ = losses.FocalTverskyLoss3D()
    elif _type == 'weighted_focal':
        criterion_ = losses.WeightedFocalTverskyLoss()
    return criterion_


def configure_optimizer(model: torch.nn.Module, optimizer: str = 'adam',
                        lr: float = 1e-4, weight_decay: float = 1e-5
                        ) -> torch.optim:
    """
    Configures and returns the NAdam optimizer for the given model.

    Parameters
    ----------
    optimizer : string
        Which optimizer to use {'adam', 'nadam}
    model : torch.nn.Module
        The model to be optimized.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay (L2 penalty). Can be None

    Returns
    -------
    torch.optim.NAdam
        Configured NAdam optimizer.
    """
    config = {
        'lr': lr
    }
    if isinstance(weight_decay, float):
        config['weight_decay'] = weight_decay

    optimizers = {
        'adam': torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay),
        #'nadam': torch.optim.NAdam(
        #    model.parameters(), lr=lr, weight_decay=weight_decay)
        }

    return optimizers[optimizer]


def main_train(
        model, train_loader, val_loader, optimizer, criterion, num_epochs=25,
        lr=0.001, model_dir='output/models', warmup_epochs=5,
        cruise_epochs=1000, cooldown_epochs=5, device='cuda'
        ):

    clear_directory_files(model_dir)
    clear_directory_files(f"{model_dir}/checkpoints")

    best_vloss = 1.0
    writer = log_model_graph(model_dir, model, train_loader)

    # Training loop
    for epoch in range(num_epochs):
        # model, epoch, writer, loader, opt, criterion
        train_one_epoch(
            model, epoch, writer, train_loader, criterion, optimizer,
            device)

        best_vloss = validate(
            model, epoch, writer, val_loader, optimizer, criterion,
            device, best_vloss, model_dir)

        log_hist(model, epoch, writer)

    torch.cuda.empty_cache()
    writer.close()
    print("Training completed and CUDA memory cleaned up")
    return model

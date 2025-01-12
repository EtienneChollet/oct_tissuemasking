from torch import multiprocessing as mp
from torch import autograd
from oct_tissuemasking.train import main_train, configure_optimizer
from oct_tissuemasking.models import ScaledUNet
from oct_tissuemasking.data import get_loaders
from oct_tissuemasking.losses import DiceLoss

if __name__ == '__main__':

    model_dir = 'output/models/version_128.1'
    data_path = '/autofs/cluster/octdata2/users/epc28/oct_tissuemasking/data/training_data_128'

    autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn')

    # Init model
    model = ScaledUNet(
        n_classes=1,
        base_filters=8,
        scale_factor=1,
        dropout=0.0
        ).to('cuda').float()
    # Get the data loaders
    train_loader, val_loader = get_loaders(
        subset=-1,
        batch_size=1,
        data_path=data_path
        )
    # Init optimizer
    optimizer = configure_optimizer(
        model=model,
        lr=1e-4,
        weight_decay=1e-12)
    # Init criterion
    criterion = DiceLoss(activation='Sigmoid')

    main_train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=1000,
        model_dir=model_dir
    )

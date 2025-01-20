import numpy as np
import nibabel as nib
from cyclopts import App
from oct_tissuemasking.config import __version__
from oct_tissuemasking.models import (
    ModelConfigManager, FullPredict, get_checkpoint_from_version
)

app = App(version=__version__)


@app.command()
def version():
    """
    Display the current version of the package.
    """
    print(f"oct_tissuemasking version: {__version__}")


# TODO: Figure out how to set req arguments without throwing error in calling.
@app.command()
def predict(in_path: str, out_path: str, model: str = None, patch_size: int = 128,
            step_size: int = 64):
    """
    Predict the OCT tissue mask on a specified volume with a specified trained
    model.

    Returns a binarized tissue mask

    Parameters
    ----------
    model : str
        Model version to test.
    in_path : str
        Path to NIfTI volume to preform the prediction on.
    out_path : str
        Path to save binarized tissue mask prediction to.
    patch_size : int
        Size of model's input layer (cube).
    step_size : int
        Size of step between adjacent prediction patches.
    """
    import torch
    # Get the full path to the model checkpoint
    checkpoint_path = str(get_checkpoint_from_version())

    # Init the model configuration manager
    config_manager = ModelConfigManager(n_classes=1, verbose=True)

    # Build the model from the checkpoint path
    model = config_manager.build_and_load_model(
        1,
        'cuda',
        checkpoint_path=checkpoint_path,
    )

    model.eval().cuda()
    print('Model Loaded...')

    # Loading nifti data from specified path
    nifti = nib.load(in_path)
    affine = nifti.affine
    in_tensor = torch.from_numpy(nifti.get_fdata()).cuda()
    print('NIfTI Loaded...')

    # Normalizing
    in_tensor -= in_tensor.min()
    in_tensor /= in_tensor.max()

    # Init prediction class with optional custom step and patch sizes
    prediction = FullPredict(
        in_tensor.to(torch.float32), model, patch_size=patch_size,
        step_size=step_size)

    # Execute prediction
    prediction.predict()
    print('Prediction Complete...')

    # Thresholding
    out_tensor = torch.clone(prediction.imprint_tensor)
    print(out_tensor.min())
    print(out_tensor.max())
    prediction.imprint_tensor[prediction.imprint_tensor < 0.5] = 0
    prediction.imprint_tensor[prediction.imprint_tensor >= 0.5] = 1
    prediction.imprint_tensor = prediction.imprint_tensor.cpu().numpy().astype(
        np.uint8
    )[0][0]

    nib.save(
        nib.nifti1.Nifti1Image(
            dataobj=prediction.imprint_tensor,
            affine=affine),
        filename=out_path)

@app.command()
def test_prediction():
    from importlib.resources import files
    model = None
    if model is None:
        checkpoint_path = files("oct_tissuemasking.checkpoints").joinpath(
            "version-1_checkpoint.tar"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Default model checkpoint not found at: {checkpoint_path}"
            )
        print('Using default model found at:', checkpoint_path)
    else:
        checkpoint_path = Path(model)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Specified model checkpoint not found: {model}")

    print(checkpoint_path)



if __name__ == '__main__':
    app()

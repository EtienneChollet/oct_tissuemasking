import numpy as np
import nibabel as nib
from cyclopts import App


app = App()


# TODO: Figure out how to set req arguments without throwing error in calling.
@app.command()
def predict(model: str, in_path: str, out_path: str, patch_size: int = 128,
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
    if not model or not in_path or not out_path:
        print("Missing arguments. Usage: command --model MODEL --in_path IN_PATH --out_path OUT_PATH")
        exit(0)
    import torch
    from oct_tissuemasking.models import ModelConfigManager, FullPredict
    # Loading model from weights
    base_model_dir = (f'/autofs/cluster/octdata2/users/epc28/oct_tissuemasking/output/models/version_{model}')
    config_manager = ModelConfigManager(base_model_dir, n_classes=1,
                                        verbose=True)
    model = config_manager.build_and_load_model(1, device='cuda')
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
    prediction.imprint_tensor = prediction.imprint_tensor.cpu().numpy().astype(np.uint8)[0][0]

    nib.save(
        nib.nifti1.Nifti1Image(
            dataobj=prediction.imprint_tensor,
            affine=affine),
        filename=out_path)


# @app.default
# def default_action():
#    print("Hello world! This runs when no command is specified.")

if __name__ == '__main__':
    app()

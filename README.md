# oct_tissuemasking

`oct_tissuemasking` is a command-line tool for generating tissue masks from 3D Optical Coherence Tomography (OCT) data. It uses lightweight, pre-trained models and patch-based predictions to output binarized tissue masks.

# Features

- **Predict Tissue Masks**: Generate binarized tissue masks for OCT volumes using a trained model.
- **Customizable Parameters**: Adjust patch size and step size for the prediction process.

# 1 Installation

## 1.1 Create a new Mamba Environment

Create a new mamba environment called `oct_tissuemasking` with python 3.9.

```bash
>>> mamba create -n oct_tissuemasking python=3.9
>>> mamba activate oct_tissuemasking
```

## 1.2 Install oct_tissuemasking from PyPi

Now we can just install the `oct_tissuemasking` package from PyPi!

```bash
>>> pip install oct_tissuemasking
```

# 2 Usage

## 2.1 Predict Tissue Masks

Use the following command to make a tissue mask from the OCT volume located at`--in-path`.

```bash
python oct_tissuemasking predict --in-path <INPUT_PATH> \
							    						--out-path <OUTPUT_PATH> \
                              [--model <MODEL_VERSION>] \ 	# Optional, default 1
                              [--patch-size <PATCH_SIZE>] \ # Optional, default 128
                              [--step-size <STEP_SIZE>]			# Optional, default 128
```

### Parameters:

- `--in-path`: Path to the input NIfTI file.
- `--out-path`: Path to save the output binarized tissue mask (NIfTI format).
- `--model` (optional): Version of the model to use. Defaults to the version specified in the package.
- `--patch-size` (optional): Size of the model input patch (default: 128).
- `--step-size` (optional): Step size between adjacent patches during prediction (default: 64).

### Example:

To generate a tissue mask:

```bash
python oct_tissuemasking predict --in-path input_volume.nii.gz \
                              --out-path output_mask.nii.gz \
                              --model 1 \
                              --patch-size 128 \
                              --step-size 64
```
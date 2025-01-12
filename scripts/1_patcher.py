import argparse
import torch
import numpy as np
import nibabel as nib
from torch import nn
from oct_tissuemasking.utils import clear_directory_files


class DataPreprocessingModule(nn.Module):
    def __init__(self, x_path: str, y_path: str, base_out_path: str,
                 patch_sz: int = 128, x_dtype=torch.float32, y_dtype=torch.uint8):
        """
        Initialize the data preprocessing module.

        Parameters
        ----------
        x_path : str
            Path to the input x NIfTI file.
        y_path : str
            Path to the input y NIfTI file.
        base_out_path : str
            Base path for saving output patches.
        patch_sz : int, optional
            Size of the patches, by default 128.
        x_dtype : torch.dtype, optional
            Data type for saving x patches, by default torch.float32.
        y_dtype : torch.dtype, optional
            Data type for saving y patches, by default torch.uint8.
        """
        super(DataPreprocessingModule, self).__init__()
        self.x_path = x_path
        self.y_path = y_path
        self.base_out_path = base_out_path
        self.patch_sz = patch_sz
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

        # Load and preprocess X
        self.x_in = nib.load(x_path).get_fdata()
        self.x_in = torch.from_numpy(self.x_in).float().cuda()
        self.x_in -= self.x_in.min()
        self.x_in /= self.x_in.max()
        print("Loaded X")

        # Load and preprocess Y
        y_nib = nib.load(y_path)
        self.affine = y_nib.affine
        self.y_in = torch.from_numpy(y_nib.get_fdata()).to(torch.uint8).cuda()
        self.y_in[self.y_in != 0] = 1
        print("Loaded Y")

        self.shape = self.y_in.shape
        self.x_coords = range(patch_sz, self.shape[0] - patch_sz, patch_sz)
        self.y_coords = range(patch_sz, self.shape[1] - patch_sz, patch_sz)
        self.z_coords = range(patch_sz, self.shape[2] - patch_sz, patch_sz)
        self.complete_patch_coords = [
            [x, y, z] for x in self.x_coords for y in self.y_coords for z in self.z_coords
        ]

    def forward(self):
        """
        Forward method for the data preprocessing module. It generates patches
        from the input data and saves them.
        """
        clear_directory_files(f'{self.base_out_path}/x')
        clear_directory_files(f'{self.base_out_path}/y')

        for idx, (x, y, z) in enumerate(self.complete_patch_coords):
            print(f"Patch {idx}")

            # Update affine matrix
            aff = np.copy(self.affine)
            M = aff[:3, :3]
            abc = aff[:3, 3]
            abc += np.diag(M * [x, y, z])

            # Get and preprocess x_out
            x_out = self.x_in[
                x:x + self.patch_sz,
                y:y + self.patch_sz,
                z:z + self.patch_sz
                ]
            if x_out.max() != 0:
                x_out /= x_out.max()
            x_out = x_out.unsqueeze(0)

            # Get and preprocess y_out
            y_out = self.y_in[
                x:x + self.patch_sz,
                y:y + self.patch_sz,
                z:z + self.patch_sz
                ]
            y_out = y_out.unsqueeze(0)

            # Save tensors
            x_outpath = f'{self.base_out_path}/x/patch-{idx}.pt'
            y_outpath = f'{self.base_out_path}/y/patch-{idx}.pt'

            torch.save(x_out.to(self.x_dtype).cpu(), x_outpath)
            torch.save(y_out.to(self.y_dtype).cpu(), y_outpath)

            # Uncomment if you want to save as NIfTI
            # nib.save(nib.Nifti1Image(dataobj=x_out.cpu().numpy(), affine=aff), x_outpath)
            # nib.save(nib.Nifti1Image(dataobj=y_out.cpu().numpy(), affine=aff), y_outpath)


def parse_args():
    parser = argparse.ArgumentParser(description='Data Preprocessing Module')
    parser.add_argument('--x_path', type=str,
                        default='/autofs/space/omega_001/users/caa/CAA26_Occipital/Process_caa26_occipital/mus/mus_mean_20um-iso.nii',
                        help='Path to the input x NIfTI file')
    parser.add_argument('--y_path', type=str,
                        default='/autofs/cluster/octdata2/users/epc28/data/CAA/caa26/occipital/caa26_occipital_mask_ERODED-5x_largest-cc_EC-Edited.nii',
                        help='Path to the input y NIfTI file')
    parser.add_argument('--base_out_path', type=str,
                        default='/autofs/cluster/octdata2/users/epc28/oct_tissuemasking/data/training_data_128',
                        help='Base path for saving output patches')
    parser.add_argument('--patch_sz', type=int, default=128,
                        help='Size of the patches, default is 128')
    parser.add_argument('--x_dtype', type=str, default='float32',
                        help='Data type for saving x patches, default is float32')
    parser.add_argument('--y_dtype', type=str, default='uint8',
                        help='Data type for saving y patches, default is uint8')
    return parser.parse_args()


def main():
    args = parse_args()

    x_dtype = getattr(torch, args.x_dtype)
    y_dtype = getattr(torch, args.y_dtype)

    module = DataPreprocessingModule(
        x_path=args.x_path,
        y_path=args.y_path,
        base_out_path=args.base_out_path,
        patch_sz=args.patch_sz,
        x_dtype=x_dtype,
        y_dtype=y_dtype
    )
    module.forward()


if __name__ == "__main__":
    main()

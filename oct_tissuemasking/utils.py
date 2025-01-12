import os
import shutil


def clear_directory_files(path: str, verbose: bool = False,
                          create_if_missing: bool = True
                          ) -> None:
    """
    Checks if the given directory has files and deletes all files within it.

    Parameters:
    path (str): The path to the directory where files will be checked and
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

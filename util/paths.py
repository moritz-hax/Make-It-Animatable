from pathlib import Path
import shutil

def path_join(*paths: str | Path) -> str:
    """Join multiple paths together.

    This function joins multiple paths together using the '/' separator.

    Arguments:
        *paths: str or Path. The paths to join.
        out (str): The output type. Default is 'str'.

    Returns:
        str or Path: The joined path.

    Example:
        >>> path_join("path", "to", "file", out="str")
        'path/to/file'

    Note:
        This function is useful in joining multiple paths together.

    """
    return "/".join([str(p) for p in paths])

def path_exists(path: str | Path) -> bool:
    """Check if a path exists.

    This function checks if the specified path exists.

    Arguments:
        path (str | Path): The path to check for existence.

    Returns:
        bool: True if the path exists, False otherwise.

    Example:
        >>> exists("path/to/file")
        True

    Note:
        This function is useful in checking if a path exists.

    """
    return Path(path).exists()

def path_cwd() -> str:
    """Get the current working directory."""
    return str(Path.cwd())


def get_blender_path(blender_path: str | None = None) -> str:
    """Verify Blender installation and return executable path.

    Args:
        blender_path: Optional explicit path to Blender executable

    Returns:
        Path to Blender executable

    Raises:
        RuntimeError: If Blender is not found

    """
    if blender_path and path_exists(blender_path):
        print(f"Using specified Blender: {blender_path}")
        return blender_path

    # Try to find Blender in system PATH
    blender_bin = shutil.which("blender")
    if blender_bin:
        print(f"Found Blender: {blender_bin}")
        return blender_bin

    error_msg = (
        "Blender not found! Please either:\n"
        "1. Install Blender and add to PATH\n"
        "2. Provide explicit path to Blender executable\n"
        "Installation: 'sudo snap install blender --classic' (Linux)\n"
        "or download from https://www.blender.org/download/"
    )
    print(error_msg)
    msg = "Blender installation not found"
    raise RuntimeError(msg)

if __name__ == "__main__":
    bpy.ops.wm.read_factory_settings(use_empty=False)
    obj_list: "list[Object]" = list(bpy.context.scene.objects)
    print(obj_list)
    obj = obj_list[0]
    print(obj)

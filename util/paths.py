from pathlib import Path

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

PATH_DATA = path_join(path_cwd(), "data")
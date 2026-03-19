"""Shared utilities for the asset transformer."""


def make_explicit_relative(rel_path: str) -> str:
    """Ensure a relative path starts with an explicit ``./`` or ``../`` prefix.

    ``os.path.relpath`` omits the ``./`` when the target is in the same
    directory or a subdirectory.  USD best practice is to always write
    explicit relative anchors so that tooling never confuses a bare
    filename with a search-path identifier.

    Args:
        rel_path: A relative path (forward or back-slash separators).

    Returns:
        The path guaranteed to start with ``./`` or ``../``.
    """
    if not rel_path:
        return rel_path
    if rel_path.startswith("./") or rel_path.startswith("../"):
        return rel_path
    if rel_path.startswith(".\\") or rel_path.startswith("..\\"):
        return rel_path
    return f"./{rel_path}"

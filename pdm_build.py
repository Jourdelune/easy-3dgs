import subprocess
from pdm.backend.hooks.base import Context


def pdm_build_initialize(context: Context):
    """
    This hook is called before the build process starts.
    """
    # Get the project root path
    project_root = context.root

    # Path to the .gitmodules file
    gitmodules_path = project_root / ".gitmodules"

    if gitmodules_path.exists():
        print("Initializing and updating Git submodules...")
        try:
            # Run 'git submodule update --init --recursive'
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                check=True,
                capture_output=True,
                text=True,
            )
            print("Git submodules initialized and updated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error initializing Git submodules: {e.stderr}")
            raise
    else:
        print(".gitmodules not found, skipping submodule initialization.")

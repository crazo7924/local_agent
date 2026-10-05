"""Tools available to the LLM agent."""

import os
import shlex
import subprocess

from src.tools.executables import list_path_executables


def list_directory(directory_path: str) -> str:
    """Lists the contents of a directory.

    Args:
        directory_path: The parent directory of which the contents are returned comma separated.
    """
    return ", ".join(os.listdir(directory_path))


def read_file(file_path: str) -> str:
    """Reads the content of a file from the local file system.

    Args:
        file_path: The absolute or relative path to the file.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(2000)  # Limit to 2k chars to prevent context overflow
            if len(content) == 2000:
                content += "\n...[truncated]..."
            return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


def launch_app(application_name: str, args: str) -> str:
    """Launches a desktop application in the Linux environment.

    Supported shortcuts: 'chrome', 'vscode', 'terminal'.

    Args:
        application_name: The name of the app to launch.
        args: Command line arguments to pass to the application.
    """
    app_map = {
        "chrome": "google-chrome",
        "google chrome": "google-chrome",
        "vscode": "code",
        "code": "code",
        "terminal": "xdg-terminal-exec",
    }

    command = app_map.get(application_name.lower(), application_name)

    try:
        parsed_args = shlex.split(args) if args else []
        cmd_list = [command] + parsed_args
        subprocess.Popen(cmd_list, shell=False, start_new_session=True)
        return f"Successfully launched {application_name} (PID: running in background)."
    except Exception as e:
        return f"Failed to launch {application_name}: {str(e)}"


available_tools = {
    "read_file": read_file,
    "launch_app": launch_app,
    "list_directory": list_directory,
    "list_path_executables": list_path_executables,
}

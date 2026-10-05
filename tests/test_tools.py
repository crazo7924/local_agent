"""Tests for tools in src/tools.py."""

from unittest.mock import MagicMock, patch
from src.tools.file_ops import launch_app


@patch("subprocess.Popen")
def test_launch_app_success(mock_popen):
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc

    result = launch_app("chrome", "--incognito https://example.com")

    assert "Successfully launched chrome" in result
    mock_popen.assert_called_once_with(
        ["google-chrome", "--incognito", "https://example.com"],
        shell=False,
        start_new_session=True,
    )


@patch("subprocess.Popen")
def test_launch_app_no_args(mock_popen):
    result = launch_app("vscode", "")

    assert "Successfully launched vscode" in result
    mock_popen.assert_called_once_with(
        ["code"],
        shell=False,
        start_new_session=True,
    )


@patch("subprocess.Popen")
def test_launch_app_command_injection_prevention(mock_popen):
    # Attempt command injection with shell metacharacters
    injection_args = "--url https://example.com; rm -rf /; echo injection"
    result = launch_app("terminal", injection_args)

    assert "Successfully launched terminal" in result
    mock_popen.assert_called_once_with(
        [
            "xdg-terminal-exec",
            "--url",
            "https://example.com;",
            "rm",
            "-rf",
            "/;",
            "echo",
            "injection",
        ],
        shell=False,
        start_new_session=True,
    )


def test_launch_app_malformed_quotes_error():
    # Unmatched quote in args causes shlex.split to raise ValueError
    result = launch_app("chrome", '--foo "bar')

    assert "Failed to launch chrome" in result
    assert "No closing quotation" in result or "No escaped character" in result


@patch("subprocess.Popen")
def test_launch_app_popen_exception(mock_popen):
    mock_popen.side_effect = FileNotFoundError(
        "No such file or directory: 'nonexistent'"
    )

    result = launch_app("nonexistent_app", "--flag")

    assert "Failed to launch nonexistent_app" in result
    assert "No such file or directory" in result

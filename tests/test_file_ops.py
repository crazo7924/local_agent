"""Tests for file_ops tools."""

from unittest.mock import patch

from src.tools.file_ops import launch_app, list_directory, read_file


def test_launch_app_shortcut_with_list_args():
    with patch("subprocess.Popen") as mock_popen:
        res = launch_app("chrome", ["--incognito", "http://example.com"])
        assert "Successfully launched chrome" in res
        mock_popen.assert_called_once_with(
            ["google-chrome", "--incognito", "http://example.com"],
            shell=False,
            start_new_session=True,
        )


def test_launch_app_shortcut_with_str_args():
    with patch("subprocess.Popen") as mock_popen:
        res = launch_app("vscode", "--new-window /path/to/project")
        assert "Successfully launched vscode" in res
        mock_popen.assert_called_once_with(
            ["code", "--new-window", "/path/to/project"],
            shell=False,
            start_new_session=True,
        )


def test_launch_app_empty_args():
    with patch("subprocess.Popen") as mock_popen:
        res = launch_app("terminal")
        assert "Successfully launched terminal" in res
        mock_popen.assert_called_once_with(
            ["xdg-terminal-exec"],
            shell=False,
            start_new_session=True,
        )


def test_launch_app_prevents_command_injection():
    with patch("subprocess.Popen") as mock_popen:
        payload = "; echo 'hacked' #"
        res = launch_app("chrome", payload)
        assert "Successfully launched chrome" in res
        mock_popen.assert_called_once_with(
            ["google-chrome", ";", "echo", "'hacked'", "#"],
            shell=False,
            start_new_session=True,
        )


def test_launch_app_exception_handling():
    with patch("subprocess.Popen", side_effect=OSError("Permission denied")):
        res = launch_app("invalid_app")
        assert "Failed to launch invalid_app: Permission denied" in res


def test_read_file_nonexistent(tmp_path):
    res = read_file(str(tmp_path / "nonexistent.txt"))
    assert "Error: File" in res


def test_read_file_existing(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello World", encoding="utf-8")
    res = read_file(str(test_file))
    assert res == "Hello World"


def test_list_directory(tmp_path):
    (tmp_path / "a.txt").touch()
    (tmp_path / "b.txt").touch()
    res = list_directory(str(tmp_path))
    assert "a.txt" in res
    assert "b.txt" in res

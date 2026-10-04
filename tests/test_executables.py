import json
import os
import unittest
from unittest.mock import patch, MagicMock

from src.tools.executables import list_path_executables, PackageManager, _query_rpm


class TestListPathExecutables(unittest.TestCase):
    def test_list_path_executables_format(self):
        result_str = list_path_executables(package_manager="none")
        result = json.loads(result_str)

        self.assertIn("managed", result)
        self.assertIn("unmanaged", result)
        self.assertIsInstance(result["managed"], list)
        self.assertIsInstance(result["unmanaged"], list)

    @patch("os.environ.get")
    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.path.isfile")
    @patch("os.access")
    def test_list_path_executables_mocked_files(
        self, mock_access, mock_isfile, mock_isdir, mock_listdir, mock_env_get
    ):
        mock_env_get.return_value = "/mock/bin"
        mock_isdir.return_value = True
        mock_listdir.return_value = ["app1", "app2"]
        mock_isfile.return_value = True
        mock_access.return_value = True

        result_str = list_path_executables(package_manager="none")
        result = json.loads(result_str)

        self.assertEqual(result["managed"], [])
        self.assertIn("/mock/bin/app1", result["unmanaged"])
        self.assertIn("/mock/bin/app2", result["unmanaged"])

    @patch("src.tools.executables._query_dpkg")
    @patch("src.tools.executables._detect_package_manager")
    def test_list_path_executables_dpkg_managed(self, mock_detect, mock_dpkg):
        mock_detect.return_value = PackageManager.DPKG
        mock_dpkg.return_value = {"/bin/ip": "iproute2"}

        with patch.dict(os.environ, {"PATH": "/bin"}):
            with patch("os.listdir", return_value=["ip"]):
                with patch("os.path.isdir", return_value=True):
                    with patch("os.path.isfile", return_value=True):
                        with patch("os.access", return_value=True):
                            result_str = list_path_executables(package_manager="auto")
                            result = json.loads(result_str)

                            self.assertEqual(
                                result["managed"],
                                [{"package": "iproute2", "binary": "/bin/ip"}],
                            )
                            self.assertEqual(result["unmanaged"], [])

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_query_rpm_parsing(self, mock_subprocess_run, mock_which):
        mock_which.return_value = "/usr/bin/rpm"
        mock_res = MagicMock()
        mock_res.stdout = "/bin/ls\tcoreutils\n/bin/unmanaged_bin\t\n"
        mock_subprocess_run.return_value = mock_res

        binaries = ["/bin/ls", "/bin/unmanaged_bin"]
        mapping = _query_rpm(binaries)

        self.assertEqual(mapping, {"/bin/ls": "coreutils"})


if __name__ == "__main__":
    unittest.main()

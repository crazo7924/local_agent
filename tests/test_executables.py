import json
import os
import unittest
from unittest.mock import MagicMock, patch

from src.tools.executables import PackageManager, _query_dnf, _query_rpm, list_path_executables


class TestListPathExecutables(unittest.TestCase):
    def setUp(self):
        list_path_executables.cache_clear()

    def tearDown(self):
        list_path_executables.cache_clear()

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

    @patch("src.tools.executables._query_dpkg")
    def test_list_path_executables_apt_option(self, mock_dpkg):
        mock_dpkg.return_value = {"/bin/ip": "iproute2"}

        with patch.dict(os.environ, {"PATH": "/bin"}):
            with patch("os.listdir", return_value=["ip"]):
                with patch("os.path.isdir", return_value=True):
                    with patch("os.path.isfile", return_value=True):
                        with patch("os.access", return_value=True):
                            result_str = list_path_executables(package_manager="apt")
                            result = json.loads(result_str)

                            self.assertEqual(
                                result["managed"],
                                [{"package": "iproute2", "binary": "/bin/ip"}],
                            )

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

    def test_caching_behavior_and_hits(self):
        with patch("os.listdir", return_value=["app1"]):
            with patch("os.path.isdir", return_value=True):
                with patch("os.path.isfile", return_value=True):
                    with patch("os.access", return_value=True):
                        res1 = list_path_executables("none")
                        res2 = list_path_executables("none")

                        self.assertEqual(res1, res2)

        info = list_path_executables.cache_info()
        self.assertGreaterEqual(info.hits, 1)

    def test_argument_case_insensitivity_shares_cache(self):
        with patch("os.listdir", return_value=["app1"]):
            with patch("os.path.isdir", return_value=True):
                with patch("os.path.isfile", return_value=True):
                    with patch("os.access", return_value=True):
                        res1 = list_path_executables("NONE")
                        res2 = list_path_executables("none")

                        self.assertEqual(res1, res2)

        info = list_path_executables.cache_info()
        self.assertGreaterEqual(info.hits, 1)

    def test_path_change_invalidates_cache(self):
        with patch("os.listdir", return_value=["app1"]):
            with patch("os.path.isdir", return_value=True):
                with patch("os.path.isfile", return_value=True):
                    with patch("os.access", return_value=True):
                        with patch.dict(os.environ, {"PATH": "/dir1"}):
                            list_path_executables("none")

                        with patch.dict(os.environ, {"PATH": "/dir2"}):
                            list_path_executables("none")

        info = list_path_executables.cache_info()
        self.assertEqual(info.misses, 2)

    def test_explicit_cache_clear(self):
        with patch("os.listdir", return_value=["app1"]):
            with patch("os.path.isdir", return_value=True):
                with patch("os.path.isfile", return_value=True):
                    with patch("os.access", return_value=True):
                        list_path_executables("none")
                        info_before = list_path_executables.cache_info()
                        self.assertEqual(info_before.currsize, 1)

                        list_path_executables.cache_clear()
                        info_after = list_path_executables.cache_info()
                        self.assertEqual(info_after.currsize, 0)
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_query_dnf_parsing(self, mock_subprocess_run, mock_which):
        def which_side_effect(cmd):
            if cmd == "rpm":
                return None
            if cmd == "dnf":
                return "/usr/bin/dnf"
            return None

        mock_which.side_effect = which_side_effect

        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "coreutils\t/bin/ls /bin/cat\n"
        mock_subprocess_run.return_value = mock_res

        binaries = ["/bin/ls", "/bin/unmanaged_bin"]
        mapping = _query_dnf(binaries, chunk_size=2)

        self.assertEqual(mapping, {"/bin/ls": "coreutils"})
        mock_subprocess_run.assert_called_once_with(
            [
                "dnf",
                "repoquery",
                "--installed",
                "--file",
                "/bin/ls",
                "--file",
                "/bin/unmanaged_bin",
                "--queryformat",
                "%{name}\t%{files}",
            ],
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()

import functools
import json
import os
import shutil
import subprocess
from enum import Enum
from typing import Dict, List, Optional


class PackageManager(str, Enum):
    AUTO = "auto"
    APT = "apt"
    DNF = "dnf"
    DPKG = "dpkg"
    RPM = "rpm"
    PACMAN = "pacman"
    NONE = "none"


def _query_dpkg(binaries: List[str]) -> Dict[str, str]:
    """Query dpkg-query -S for a list of binaries."""
    mapping = {}
    if not shutil.which("dpkg-query") or not binaries:
        return mapping

    # dpkg-query -S binary1 binary2 ...
    # Output format: "package: /path/to/binary"
    chunk_size = 500
    for i in range(0, len(binaries), chunk_size):
        chunk = binaries[i : i + chunk_size]
        try:
            res = subprocess.run(
                ["dpkg-query", "-S"] + chunk,
                capture_output=True,
                text=True,
            )
            for line in res.stdout.splitlines():
                if ":" in line:
                    parts = line.split(":", 1)
                    pkg = parts[0].strip()
                    path = parts[1].strip()
                    mapping[path] = pkg
        except Exception:
            pass
    return mapping


def _query_rpm(binaries: List[str]) -> Dict[str, str]:
    """Query rpm -qf for binaries."""
    mapping = {}
    if not shutil.which("rpm") or not binaries:
        return mapping

    chunk_size = 500
    for i in range(0, len(binaries), chunk_size):
        chunk = binaries[i : i + chunk_size]
        try:
            res = subprocess.run(
                ["rpm", "-qf", "--queryformat", "%{FILENAMES}\t%{NAME}\n"] + chunk,
                capture_output=True,
                text=True,
            )
            for line in res.stdout.splitlines():
                if "\t" in line:
                    path, pkg_name = line.split("\t", 1)
                    path = path.strip()
                    pkg_name = pkg_name.strip()
                    if path and pkg_name:
                        mapping[path] = pkg_name
        except Exception:
            pass
    return mapping


def _query_dnf(binaries: List[str], chunk_size: int = 500) -> Dict[str, str]:
    """Query package manager for RPM-based systems (using fast rpm query first if available)."""
    if shutil.which("rpm"):
        return _query_rpm(binaries)

    mapping = {}
    if not shutil.which("dnf") or not binaries:
        return mapping

    for i in range(0, len(binaries), chunk_size):
        chunk = binaries[i : i + chunk_size]
        chunk_set = set(chunk)
        cmd = ["dnf", "repoquery", "--installed"]
        for bin_path in chunk:
            cmd.extend(["--file", bin_path])
        cmd.extend(["--queryformat", "%{name}\t%{files}"])

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if res.returncode == 0 and res.stdout:
                current_pkg = None
                for line in res.stdout.splitlines():
                    line_str = line.strip()
                    if not line_str:
                        continue
                    if "\t" in line_str:
                        parts = line_str.split("\t", 1)
                        current_pkg = parts[0].strip()
                        rest = parts[1].strip()
                        for token in rest.split():
                            if token in chunk_set and current_pkg:
                                mapping[token] = current_pkg
                    elif current_pkg:
                        for token in line_str.split():
                            if token in chunk_set:
                                mapping[token] = current_pkg
        except Exception:
            pass
    return mapping


def _query_pacman(binaries: List[str]) -> Dict[str, str]:
    """Query pacman -Qo for binaries."""
    mapping = {}
    if not shutil.which("pacman") or not binaries:
        return mapping

    chunk_size = 500
    for i in range(0, len(binaries), chunk_size):
        chunk = binaries[i : i + chunk_size]
        try:
            res = subprocess.run(
                ["pacman", "-Qo"] + chunk,
                capture_output=True,
                text=True,
            )
            for line in res.stdout.splitlines():
                if " is owned by " in line:
                    path_part, pkg_part = line.split(" is owned by ", 1)
                    pkg_name = pkg_part.split()[0]
                    mapping[path_part.strip()] = pkg_name
        except Exception:
            pass
    return mapping


def _detect_package_manager() -> Optional[PackageManager]:
    if shutil.which("dpkg-query"):
        return PackageManager.DPKG
    if shutil.which("dnf"):
        return PackageManager.DNF
    if shutil.which("rpm"):
        return PackageManager.RPM
    if shutil.which("pacman"):
        return PackageManager.PACMAN
    return None


@functools.cache
def _list_path_executables_cached(package_manager_str: str, path_env: str) -> str:
    pm_enum = PackageManager(package_manager_str)

    path_dirs = [d for d in path_env.split(os.pathsep) if d and os.path.isdir(d)]

    executables_set = set()
    for directory in path_dirs:
        try:
            for entry in os.listdir(directory):
                full_path = os.path.join(directory, entry)
                if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                    executables_set.add(os.path.abspath(full_path))
        except OSError:
            continue

    sorted_executables = sorted(executables_set)

    if pm_enum == PackageManager.AUTO:
        effective_pm = _detect_package_manager()
    elif pm_enum == PackageManager.NONE:
        effective_pm = None
    else:
        effective_pm = pm_enum

    mapping = {}
    if effective_pm in (PackageManager.DPKG, PackageManager.APT):
        mapping = _query_dpkg(sorted_executables)
    elif effective_pm == PackageManager.DNF:
        mapping = _query_dnf(sorted_executables)
    elif effective_pm == PackageManager.RPM:
        mapping = _query_rpm(sorted_executables)
    elif effective_pm == PackageManager.PACMAN:
        mapping = _query_pacman(sorted_executables)

    managed = []
    unmanaged = []

    for binary in sorted_executables:
        if binary in mapping:
            managed.append({"package": mapping[binary], "binary": binary})
        else:
            unmanaged.append(binary)

    output = {
        "managed": managed,
        "unmanaged": unmanaged,
    }

    return json.dumps(output, indent=2)


def list_path_executables(package_manager: str = "auto") -> str:
    """Exposes executables in the system PATH variable in a machine readable JSON format,

    categorizing binaries into managed (installed via package manager) and unmanaged files.

    Args:
        package_manager: Package manager to query ("auto", "apt", "dnf", "dpkg", "rpm", "pacman", or "none").

    Returns:
        JSON string containing "managed" list of {"package": ..., "binary": ...} and "unmanaged" list of binary paths.
    """
    try:
        pm_enum = PackageManager(package_manager.lower())
    except (ValueError, AttributeError):
        pm_enum = PackageManager.AUTO

    path_env = os.environ.get("PATH", "")
    return _list_path_executables_cached(pm_enum.value, path_env)


list_path_executables.cache_clear = _list_path_executables_cached.cache_clear  # type: ignore[attr-defined]
list_path_executables.cache_info = _list_path_executables_cached.cache_info  # type: ignore[attr-defined]

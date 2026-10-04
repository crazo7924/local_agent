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


def _query_dnf(binaries: List[str]) -> Dict[str, str]:
    """Query package manager for RPM-based systems (using fast rpm query first if available)."""
    if shutil.which("rpm"):
        return _query_rpm(binaries)

    mapping = {}
    if not shutil.which("dnf") or not binaries:
        return mapping

    chunk_size = 50
    for i in range(0, len(binaries), chunk_size):
        chunk = binaries[i : i + chunk_size]
        for bin_path in chunk:
            try:
                res = subprocess.run(
                    [
                        "dnf",
                        "repoquery",
                        "--installed",
                        "--file",
                        bin_path,
                        "--queryformat",
                        "%{name}",
                    ],
                    capture_output=True,
                    text=True,
                )
                pkg = res.stdout.strip()
                if res.returncode == 0 and pkg:
                    mapping[bin_path] = pkg
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
    except ValueError:
        pm_enum = PackageManager.AUTO

    path_env = os.environ.get("PATH", "")
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

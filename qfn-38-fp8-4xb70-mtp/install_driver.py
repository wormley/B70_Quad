#!/usr/bin/env python3
"""Offline installation of hash-locked GPU userspace packages INSIDE the image.

No apt update, network downloads, dependency-force flags, kernel changes, or debug
workarounds. Simulate first and reject any unapproved package removals/installs.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OBSOLETE = {"intel-gpu-compute", "libigc2", "libigdfcl2", "libze-intel-gpu-raytracing", "intel-media-va-driver-non-free"}


def run(command, capture=False):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def installed(name):
    result = subprocess.run(["dpkg-query", "-W", "-f=${Status}", name],
                            text=True, capture_output=True)
    return result.returncode == 0 and result.stdout.strip() == "install ok installed"


def validate_plan(text, expected):
    for line in text.splitlines():
        words = line.split()
        if len(words) < 2:
            continue
        name = words[1].split(":", 1)[0]
        if words[0] == "Remv" and name not in OBSOLETE:
            raise RuntimeError(f"Refusing unexpected package removal: {line}")
        if words[0] == "Inst" and name not in expected:
            raise RuntimeError(f"Refusing unpinned dependency installation: {line}")


def main():
    package_dir = Path(sys.argv[1])
    lock = json.loads(Path(sys.argv[2]).read_text())
    files = []
    for item in lock["artifacts"]:
        path = package_dir / Path(item["path"]).name
        if digest(path) != item["sha256"]:
            raise RuntimeError(f"Driver package SHA256 mismatch: {path}")
        for field, expected in (("Package", item["package"]), ("Version", item["version"]),
                                ("Architecture", "amd64")):
            actual = run(["dpkg-deb", "-f", path, field], capture=True).stdout.strip()
            if actual != expected:
                raise RuntimeError(f"Driver package {field} mismatch: {path}: {actual}")
        files.append(str(path))
    remove = [name + "-" for name in sorted(OBSOLETE) if installed(name)]
    # The build/run network is disabled. --no-download is not used here: this
    # apt version mishandles local .debs under that flag. Only SHA-verified local
    # files are requested and the simulated plan rejects unpinned dependencies.
    common = ["apt-get", "--no-install-recommends", "--allow-downgrades", "install"]
    plan = run([*common[:1], "--simulate", *common[1:], *files, *remove], capture=True).stdout
    print(plan, flush=True)
    validate_plan(plan, set(lock["expected_packages"]))
    run([*common[:1], "--yes", *common[1:], *files, *remove])
    for name, version in lock["expected_packages"].items():
        actual = run(["dpkg-query", "-W", "-f=${Version}", name], capture=True).stdout.strip()
        if actual != version:
            raise RuntimeError(f"Installed GPU package mismatch: {name}: {actual} != {version}")
    if any(installed(name) for name in OBSOLETE):
        raise RuntimeError("Obsolete/conflicting GPU package remains installed")
    run(["ldconfig"])
    audit = run(["dpkg", "--audit"], capture=True).stdout
    if audit.strip():
        raise RuntimeError(f"Unconfigured/broken dpkg state: {audit}")
    # Remove volatile build logs/caches from this layer; no host files are touched.
    for directory in (Path("/var/log/apt"), Path("/var/cache/apt/archives")):
        for path in directory.glob("*"):
            if path.is_file():
                path.unlink()
    for path in (Path("/var/log/dpkg.log"), Path("/var/log/alternatives.log"),
                 Path("/var/cache/ldconfig/aux-cache")):
        path.unlink(missing_ok=True)
    print("Pinned compute-runtime/IGC/gmmlib packages installed and audited offline.")


if __name__ == "__main__":
    main()

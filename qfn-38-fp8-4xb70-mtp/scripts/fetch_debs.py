#!/usr/bin/env python3
"""Fetch any missing GPU driver debs pinned in driver-packages.lock.json.

- Files already present in driver-debs/ are hash-verified and used as-is
  (offline builds work when the debs are carried).
- Missing files are downloaded from the pinned upstream URL and verified.
- Any SHA-256 mismatch is fatal, whether the file was downloaded or carried.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent.parent
lock = json.loads((root / "driver-packages.lock.json").read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    fail = False
    for item in lock["artifacts"]:
        dest = root / item["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        have = dest.exists() and dest.stat().st_size > 0 and sha256(dest) == item["sha256"]
        if not have:
            print(f"fetching {item['package']} <- {item['url']}", flush=True)
            try:
                urllib.request.urlretrieve(item["url"], dest)
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED: {e}", flush=True)
                print("  (network is required to fetch missing driver debs; "
                      "alternatively place the file at " + str(dest) + ")", flush=True)
                fail = True
                continue
        actual = sha256(dest)
        if actual != item["sha256"]:
            print(f"  SHA256 MISMATCH for {dest.name}: {actual}", flush=True)
            fail = True
        else:
            print(f"  ok {dest.name}", flush=True)
    if fail:
        print("driver deb fetch/verify FAILED", flush=True)
        return 1
    print("all driver debs verified", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

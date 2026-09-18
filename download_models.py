"""
Downloads the two pretrained Caffe weight files (~45 MB each):

    models/age_net.caffemodel
    models/gender_net.caffemodel

Source: the original authors' repo (Gil Levi & Tal Hassner), hosted on GitHub.

Run this once before using main.py:

    python download_models.py
"""

import hashlib
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

BASE = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models"

# (filename, url, expected_size_bytes, expected_sha256)
FILES = [
    (
        "age_net.caffemodel",
        f"{BASE}/age_net.caffemodel",
        45661480,
        "6dde5d07df5ca1d66ff39e525693f05ccfb9d2c437e188fdd1a10d42e57fabd6",
    ),
    (
        "gender_net.caffemodel",
        f"{BASE}/gender_net.caffemodel",
        45649168,
        "ac7571b281ae078817764b645a20541bd6aa1babeac20a45e6d8de7d61ba0e50",
    ),
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(filename, url, expected_size, expected_sha):
    dest = os.path.join(MODELS_DIR, filename)

    if os.path.exists(dest) and os.path.getsize(dest) == expected_size:
        print(f"[skip] {filename} already present and correct size.")
        return True

    print(f"[..] Downloading {filename} ({expected_size / 1e6:.0f} MB)...")

    def progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 / total_size)
            print(f"\r     {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()

    actual_size = os.path.getsize(dest)
    if actual_size != expected_size:
        print(f"[FAIL] {filename}: expected {expected_size} bytes, got {actual_size}.")
        print("       The download was incomplete or blocked. Delete the file and retry.")
        return False

    print("     verifying checksum...")
    actual_sha = sha256_of(dest)
    if actual_sha != expected_sha:
        print(f"[FAIL] {filename}: checksum mismatch — file is corrupt.")
        return False

    print(f"[ok] {filename} downloaded and verified.")
    return True


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    all_ok = True
    for filename, url, size, sha in FILES:
        try:
            if not download(filename, url, size, sha):
                all_ok = False
        except Exception as e:
            all_ok = False
            print(f"[FAIL] Could not download {filename}: {e}")
            print(f"       Download it manually from:\n       {url}")
            print(f"       and save it as models/{filename}")

    if all_ok:
        print("\nAll models ready. Now run:  python main.py")
    else:
        print("\nSome downloads failed — see the messages above.")


if __name__ == "__main__":
    main()

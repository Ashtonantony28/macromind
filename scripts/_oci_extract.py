"""Extract selected paths from an OCI/Docker image without a docker daemon.

Usage: python3 -I _oci_extract.py <registry> <repo> <tag> <outdir> <path-prefix>...

Pulls the linux/amd64 image manifest from <registry> (anonymous, or a bearer
token if the registry asks for one), streams each layer in order and extracts
only members under the given path prefixes, applying OCI whiteouts.  Used by
setup_toolchain.sh when docker is unavailable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tarfile
import urllib.request

ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])
_token: str | None = None


def _open(url: str, accept: str | None = None):
    global _token
    for _ in range(2):
        req = urllib.request.Request(url)
        if accept:
            req.add_header("Accept", accept)
        if _token:
            req.add_header("Authorization", f"Bearer {_token}")
        try:
            return urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            hdr = e.headers.get("WWW-Authenticate", "")
            if e.code != 401 or not hdr.startswith("Bearer") or _token:
                raise
            kv = dict(re.findall(r'(\w+)="([^"]*)"', hdr))
            turl = f"{kv['realm']}?service={kv.get('service', '')}&scope={kv.get('scope', '')}"
            _token = json.load(urllib.request.urlopen(turl, timeout=60))["token"]
    raise RuntimeError("auth failed")


def _safe(name: str) -> str | None:
    name = name.lstrip("./")
    if name.startswith("/") or ".." in name.split("/"):
        return None
    return name


def main() -> int:
    reg, repo, tag, out, *prefixes = sys.argv[1:]
    prefixes = [p.strip("/") for p in prefixes]
    base = f"https://{reg}/v2/{repo}"
    man = json.load(_open(f"{base}/manifests/{tag}", ACCEPT))
    if "manifests" in man:  # index -> pick linux/amd64
        d = next(m["digest"] for m in man["manifests"]
                 if m.get("platform", {}).get("architecture") == "amd64"
                 and m.get("platform", {}).get("os") == "linux")
        man = json.load(_open(f"{base}/manifests/{d}", ACCEPT))
    os.makedirs(out, exist_ok=True)

    def wanted(n: str) -> bool:
        return any(n == p or n.startswith(p + "/") or p.startswith(n + "/") for p in prefixes)

    for i, layer in enumerate(man["layers"], 1):
        print(f"layer {i}/{len(man['layers'])} {layer['digest'][:19]} {layer['size'] / 1e6:.0f} MB", flush=True)
        with _open(f"{base}/blobs/{layer['digest']}") as resp, tarfile.open(fileobj=resp, mode="r|*") as tf:
            for m in tf:
                n = _safe(m.name)
                if not n or not wanted(n):
                    continue
                d, b = os.path.split(n)
                if b == ".wh..wh..opq":
                    p = os.path.join(out, d)
                    if os.path.isdir(p):
                        for c in os.listdir(p):
                            cp = os.path.join(p, c)
                            shutil.rmtree(cp) if os.path.isdir(cp) and not os.path.islink(cp) else os.remove(cp)
                    continue
                if b.startswith(".wh."):
                    p = os.path.join(out, d, b[4:])
                    if os.path.isdir(p) and not os.path.islink(p):
                        shutil.rmtree(p)
                    elif os.path.lexists(p):
                        os.remove(p)
                    continue
                if not any(n == p or n.startswith(p + "/") for p in prefixes) and not m.isdir():
                    continue
                if m.islnk():  # hard link inside the image
                    src = _safe(m.linkname)
                    dst = os.path.join(out, n)
                    if src and os.path.exists(os.path.join(out, src)):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        if os.path.lexists(dst):
                            os.remove(dst)
                        shutil.copy2(os.path.join(out, src), dst)
                    continue
                m.name = n
                target = os.path.join(out, n)
                if os.path.lexists(target) and not (m.isdir() and os.path.isdir(target)):
                    if os.path.isdir(target) and not os.path.islink(target):
                        shutil.rmtree(target)
                    else:
                        os.remove(target)
                tf.extract(m, out, set_attrs=True, filter="tar")
    print("extracted to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

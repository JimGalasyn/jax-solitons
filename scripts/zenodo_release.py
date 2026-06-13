#!/usr/bin/env python3
"""Mint a Zenodo DOI for a tagged jax-solitons release (stdlib only).

Usage:
    python scripts/zenodo_release.py vX.Y.Z [--sandbox] [--no-publish]

Creates a Zenodo deposition, uploads a `git archive` source tarball of the tag,
attaches metadata, and publishes — printing the concept + version DOIs. The git
tag must already exist (run after `gh release create`); the script fetches tags.

PUBLISHING IS PERMANENT: a published Zenodo record cannot be deleted, only
superseded with a new version. Use --no-publish to leave a reviewable draft.

Token: $ZENODO_TOKEN or ~/.zenodo_token (production zenodo.org by default;
--sandbox uses sandbox.zenodo.org, which needs a *separate* sandbox token).
"""
import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

META = {
    "title": "jax-solitons: a general JAX engine for classical field-theory solitons",
    "upload_type": "software",
    "license": "MIT",
    "creators": [
        {"name": "Galasyn, James P."},
        {"name": "Théodore, Claude"},
    ],
    "description": (
        "A differentiable, composable JAX engine for classical field-theory "
        "solitons — Faddeev-Skyrme hopfions, Gross-Pitaevskii vortex knots, and "
        "related topological-soliton models — with exactly-quantized "
        "(Berg-Lüscher area-form) topological charge and registered, restartable "
        "runs designed for GPU-farm-scale campaigns."
    ),
    "keywords": ["solitons", "hopfions", "skyrmions", "faddeev-skyrme",
                 "gross-pitaevskii", "topological-charge", "jax", "gpu"],
}


def read_token() -> str:
    import os
    if os.environ.get("ZENODO_TOKEN"):
        return os.environ["ZENODO_TOKEN"].strip()
    fp = Path("~/.zenodo_token").expanduser()
    if not fp.exists():
        sys.exit("no Zenodo token: set $ZENODO_TOKEN or write it to ~/.zenodo_token")
    return fp.read_text().strip()


def req(method, url, token, data=None, raw=False, ctype=None):
    headers = {"Authorization": f"Bearer {token}"}
    body = data if raw else (json.dumps(data).encode() if data is not None else None)
    if ctype:
        headers["Content-Type"] = ctype
    elif data is not None and not raw:
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or "{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="git tag, e.g. v0.0.1")
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()
    tag = args.tag
    ver = tag.lstrip("v")
    api = ("https://sandbox.zenodo.org/api" if args.sandbox
           else "https://zenodo.org/api")
    token = read_token()

    # auth precheck
    st, _ = req("GET", f"{api}/deposit/depositions?size=1", token)
    if st != 200:
        sys.exit(f"auth precheck failed: HTTP {st} (wrong token for "
                 f"{'sandbox' if args.sandbox else 'production'}?)")

    # build the source tarball from the tag
    subprocess.run(["git", "fetch", "origin", "--tags", "-q"], check=True)
    tarball = f"/tmp/jax-solitons-{ver}.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz",
                    f"--prefix=jax-solitons-{ver}/", tag, "-o", tarball], check=True)
    data = Path(tarball).read_bytes()
    print(f"archived {tag} -> {tarball} ({len(data)} bytes)")

    # 1. draft deposition
    st, dep = req("POST", f"{api}/deposit/depositions", token, data={})
    if st not in (200, 201):
        sys.exit(f"create deposition failed: HTTP {st}: {json.dumps(dep)[:300]}")
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    print(f"draft {dep_id}  pre-reserved {dep['metadata']['prereserve_doi']['doi']}")

    # 2. upload to the bucket
    st, up = req("PUT", f"{bucket}/jax-solitons-{ver}.tar.gz", token, data=data,
                 raw=True, ctype="application/octet-stream")
    if st not in (200, 201):
        sys.exit(f"upload failed: HTTP {st}: {json.dumps(up)[:300]}")
    print(f"uploaded  checksum {up.get('checksum')}")

    # 3. metadata (version + link back to the GitHub tag)
    meta = dict(META, version=ver, related_identifiers=[{
        "identifier": f"https://github.com/JimGalasyn/jax-solitons/tree/{tag}",
        "relation": "isSupplementTo", "scheme": "url"}])
    st, md = req("PUT", f"{api}/deposit/depositions/{dep_id}", token,
                 data={"metadata": meta})
    if st != 200:
        sys.exit(f"metadata failed: HTTP {st}: {json.dumps(md, indent=2)[:400]}")
    print("metadata attached")

    if args.no_publish:
        print(f"DRAFT READY (not published): {api}/deposit/depositions/{dep_id}")
        return

    # 4. publish (PERMANENT)
    st, pub = req("POST", f"{api}/deposit/depositions/{dep_id}/actions/publish",
                  token)
    if st not in (200, 202):
        sys.exit(f"PUBLISH FAILED {st}: {json.dumps(pub, indent=2)}")
    print("PUBLISHED")
    print(f"  version DOI : {pub.get('doi')}")
    print(f"  concept DOI : {pub.get('conceptdoi')}   (use for the badge)")
    print(f"  record      : {pub.get('links', {}).get('record_html')}")


if __name__ == "__main__":
    main()

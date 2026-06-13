#!/usr/bin/env python3
"""Mint a Zenodo DOI for a tagged jax-solitons release (stdlib only).

Usage:
    # first release only (mints the concept DOI):
    python scripts/zenodo_release.py v0.0.1 --first-release
    # every release after that (new version under the SAME concept):
    python scripts/zenodo_release.py vX.Y.Z --new-version-of <latest_record_id>
    #   [--sandbox] [--no-publish]

Uploads a `git archive` source tarball of the tag, attaches metadata, and
publishes — printing the concept + version DOIs. The git tag must already exist
(run after `gh release create`); the script fetches tags.

IMPORTANT: after the first release you MUST pass --new-version-of, otherwise a
fresh deposition forks a NEW concept DOI and the badge/CITATION concept DOI stops
resolving to the latest release. The script refuses to mint a new concept while
CITATION.cff already declares one (unless you pass --first-release).

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


def concept_doi_from_citation() -> str | None:
    """The concept DOI in CITATION.cff's top-level ``doi:`` field, or None.

    Used to refuse an accidental new-concept deposition: once a concept exists,
    every release must be a new VERSION under it (``--new-version-of``), or the
    concept DOI silently forks and stops resolving to the latest release.
    """
    cff = Path(__file__).resolve().parent.parent / "CITATION.cff"
    if not cff.exists():
        return None
    import re
    for line in cff.read_text().splitlines():
        m = re.match(r"doi:\s*(\S+)", line)  # top-level (col 0); identifiers use `value:`
        if m:
            return m.group(1).strip().strip("\"'")
    return None


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
    ap.add_argument("--new-version-of", type=int, metavar="RECORD_ID", default=None,
                    help="create a new VERSION under an existing record's concept "
                         "(pass the LATEST published version's record id). Required "
                         "for every release after the first so the concept DOI is "
                         "preserved instead of forking a new one. Look it up:\n"
                         "  curl -s https://zenodo.org/api/records/<conceptrecid> "
                         "| python3 -c \"import json,sys;print(json.load(sys.stdin)['id'])\"")
    ap.add_argument("--first-release", action="store_true",
                    help="intentionally mint a NEW concept DOI (the very first "
                         "release only). Refused if CITATION.cff already has one.")
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

    # 1. draft deposition — a NEW VERSION under the existing concept (preferred),
    #    or a fresh record (first release only). A fresh record mints a NEW concept
    #    DOI every time, which forks the concept and breaks "concept resolves to
    #    latest" — so we guard against doing that by accident.
    if args.new_version_of:
        st, nv = req("POST",
                     f"{api}/deposit/depositions/{args.new_version_of}/actions/newversion",
                     token)
        if st not in (200, 201, 202):
            sys.exit(f"newversion failed: HTTP {st}: {json.dumps(nv)[:300]}")
        st, dep = req("GET", nv["links"]["latest_draft"], token)
        if st != 200:
            sys.exit(f"fetch new-version draft failed: HTTP {st}: {json.dumps(dep)[:300]}")
        dep_id, bucket = dep["id"], dep["links"]["bucket"]
        for f in dep.get("files", []):  # drop inherited files; keep only this tarball
            req("DELETE", f"{api}/deposit/depositions/{dep_id}/files/{f['id']}", token)
        print(f"new-version draft {dep_id} (under the concept of record {args.new_version_of})")
    else:
        existing = concept_doi_from_citation()
        if existing and not args.first_release:
            sys.exit(
                f"refusing to mint a NEW concept: CITATION.cff already declares concept "
                f"DOI {existing}.\nPass --new-version-of <latest record id> to add a "
                f"version under it, or --first-release to fork a new concept on purpose.")
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

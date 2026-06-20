# Releasing `jax-solitons`

The maintainer runbook for cutting a tagged, Zenodo-archived release. (How to
*land a change* is the ordinary PR flow; this is the orthogonal "how to ship a
version" process.) Written from the **v0.0.1** release so it's accurate,
including the gotchas that bit us.

While the project is **pre-1.0 (0.0.x, pre-alpha)** the API is unstable — bump
the patch (`0.0.N`) for now; move to `0.1.0` when the campaign contract freezes.

## Key facts (read once)

- **The version is static, in THREE places** — bump all together (no
  setuptools-scm here):
  1. `pyproject.toml` → `version = "X.Y.Z"`
  2. `src/jax_solitons/__init__.py` → `__version__ = "X.Y.Z"`
  3. `CITATION.cff` → `version:` **and** `date-released:`
- **`main` is ruleset-protected.** A direct `git push origin main` is rejected
  (`GH013: Repository rule violations`). *Every* change — release prep and the
  DOI backfill — goes through a **PR**. Required checks: `test (3.10)` +
  `test (3.12)` + **CodeQL**; `codecov` is **non-required** (a red codecov leaves
  the PR `MERGEABLE/UNSTABLE`, still mergeable). CodeQL must exist
  (`.github/workflows/codeql.yml`) or merges wait forever on a check that never
  reports.
- **Zenodo archives the DOI via the GitHub-Release webhook** (enabled at
  zenodo.org → Account → GitHub → `jax-solitons`). Publishing a GitHub Release
  auto-creates and publishes a Zenodo record — no script, no token. Metadata
  comes from **[`.zenodo.json`](../.zenodo.json)** in the repo root (authors,
  license, keywords, abstract); keep it in sync with `CITATION.cff`. **Publishing
  is permanent** — a published record can't be deleted, only superseded.
  - **DO NOT also run `scripts/zenodo_release.py` while the webhook is enabled** —
    you'd get the release **double-archived** under two different concept DOIs.
    The script is kept only as a manual fallback (e.g. webhook outage); it now
    archives under the *legacy* concept `…20680195` and is not the normal path.
  - **The webhook mints its OWN concept DOI** the first time it archives a release
    (it can't adopt the legacy `…20680195` from the old REST-API flow). As of
    **v0.0.6** the live concept DOI is the webhook's; **v0.0.1–v0.0.5** remain
    under the legacy concept `…20680195` (their version DOIs still resolve).
- **PyPI publishes automatically** from the published GitHub Release via trusted
  publishing (OIDC, no token — `.github/workflows/publish-pypi.yml`). The built
  version comes from the static version fields, so make sure they match the tag.
  A one-time *pending publisher* registration is needed before the first publish
  (see step 3). `nwt-substrate` must be on PyPI first (the `oracle` extra now
  depends on it as a normal version specifier).
- The **concept DOI** resolves to the latest version and never changes (it's the
  README badge + `CITATION.cff` top-level `doi:`); each version gets its own
  version DOI. The current concept DOI is the **webhook's** (recorded in
  `CITATION.cff`); the legacy REST-API concept `10.5281/zenodo.20680195` covers
  v0.0.1–v0.0.5 only. The **release badge uses `?include_prereleases`** because
  0.0.x are GitHub *pre-releases*.
- **Zenodo + pre-releases:** confirm the webhook actually archives GitHub
  *pre-releases* (it has historically skipped them in some configs). After the
  first 0.0.x release post-switch, verify a record appeared
  (`curl -s -H "Authorization: Bearer $(cat ~/.zenodo_token)" "https://zenodo.org/api/deposit/depositions?q=jax-solitons&size=5"`);
  if nothing shows, the fallback is a full (non-pre) release or the manual script.
- The **Codecov *upload step* in CI needs the `CODECOV_TOKEN`** repo secret
  (already set) — `ci.yml` passes it to `codecov/codecov-action`; without it the
  upload (and hence the badge/coverage) won't update.

## Steps

### 1. Decide it's release-worthy & pick the version

```bash
git log "$(git describe --tags --abbrev=0)"..main --oneline   # what's unreleased
gh run list --branch main --workflow CI --limit 1            # main CI must be green
```

### 2. Prep PR — bump the version (no DOI yet)

Branch `release/vX.Y.Z`. Bump the **three** version locations above. There is no
`CHANGELOG.md` in this repo yet; if you keep release notes, write them for the
GitHub Release in step 3. Touch `README.md` only where it goes stale (the
`Status` line; the test count if you cite one — get it from CI:
`gh run view $(gh run list --branch main --workflow CI -L1 --json databaseId -q '.[0].databaseId') --log | grep -oE '[0-9]+ passed.*' | tail -1`).

Open the PR, let CI go green, **merge** (the ruleset blocks direct pushes).

### 3. Tag + publish the GitHub Release (pre-release for 0.0.x)

`gh` creates the tag server-side as part of the Release:

```bash
git checkout main && git pull --ff-only
gh release create vX.Y.Z --target main --prerelease \
  --title "vX.Y.Z — <one-line theme>" \
  --notes "<release notes: headline items, validation, test counts>"
```

Drop `--prerelease` (and the `?include_prereleases` on the badge) once you ship
≥ `0.1.0`.

Publishing the Release also triggers `publish-pypi.yml` (build → `twine check` →
PyPI via trusted publishing). Confirm: `gh run list --workflow publish-pypi.yml -L1`
and `curl -s https://pypi.org/pypi/jax-solitons/json -o /dev/null -w '%{http_code}\n'`.

> **One-time PyPI setup (first publish only):** register a *pending publisher* on
> pypi.org (Account → Publishing) — Project `jax-solitons`, Owner `JimGalasyn`,
> Repo `jax-solitons`, Workflow `publish-pypi.yml`, Environment `pypi` — and create
> a GitHub Environment named `pypi`.

### 4. Let the webhook archive it — then grab the DOIs

Publishing the GitHub Release (step 3) fires the Zenodo webhook automatically: it
snapshots the tag's source tarball, applies `.zenodo.json` metadata, and
**publishes** a new version record (permanent). Nothing to run — just **verify and
record the DOIs**:

```bash
# the new version record (newest first); grab its concept + version DOIs:
curl -s -H "Authorization: Bearer $(cat ~/.zenodo_token)" \
  "https://zenodo.org/api/deposit/depositions?q=jax-solitons&sort=mostrecent&size=5" \
  | python -c 'import sys,json; [print(d["metadata"].get("version"), d.get("conceptdoi"), d.get("doi")) for d in json.load(sys.stdin)]'
```

The **concept DOI** (resolves to latest) and the new **version DOI** are what you
backfill in step 5. If no new record appears within a few minutes, see the
"Zenodo + pre-releases" key-fact above — the webhook may be skipping pre-releases.

> **Fallback only (webhook outage):** `python scripts/zenodo_release.py vX.Y.Z
> --new-version-of <latest_record_id>` archives under the *legacy* concept
> `…20680195`. Do **not** run it on a release the webhook already archived —
> that double-archives the version under two concepts.

### 5. Backfill the DOI

On a `chore/backfill-vX.Y.Z-doi` branch:

- `README.md` — the **DOI badge** points at the *concept* DOI
  (`zenodo.org/badge/DOI/10.5281/zenodo.20680195.svg`); it does **not** change
  between versions, so usually no edit unless this is the first release.
- `CITATION.cff` — set top-level `doi:` (concept) and prepend the new version to
  `identifiers:` (concept + version DOIs, most-recent first).

PR, CI green, merge. Optionally refresh the Release body:
`gh release edit vX.Y.Z --notes "…"`.

## Gotchas (these bit us on v0.0.1)

- **Direct push to `main` → `GH013 Repository rule violations`.** Always
  branch + PR, even for a one-line badge change.
- **`git archive vX.Y.Z` fails** until you `git fetch origin --tags` — `gh
  release create` makes the tag *remotely*, so your local clone doesn't have it
  yet. (`zenodo_release.py` fetches for you.)
- **Zenodo publish is irreversible.** The webhook publishes automatically on
  Release, so get `.zenodo.json` right *before* you cut the Release — there's no
  draft-review step in the webhook path (only superseding fixes a bad record).
  The fallback script's `--no-publish` still leaves a reviewable draft.
- **codecov is non-required**, so a red codecov on a release/docs PR does not
  block the merge (it shows `UNSTABLE`, which is still mergeable).
- **CodeQL must report.** If the run is missing, merges hang on a never-arriving
  check; ensure `.github/workflows/codeql.yml` exists.
- Two GitHub-Actions cutovers we already handled but worth re-checking yearly:
  Node-20 → Node-24 action majors (`checkout@v5`, `setup-python@v6`).

## Quick checklist

- [ ] `main` CI green
- [ ] version bumped in `pyproject.toml` + `src/jax_solitons/__init__.py` +
      `CITATION.cff` (PR merged)
- [ ] (first publish only) PyPI pending publisher + `pypi` GitHub Environment registered
- [ ] `gh release create vX.Y.Z --prerelease` (tag + Release → triggers PyPI)
- [ ] PyPI publish workflow green; package resolves on pypi.org
- [ ] webhook archived the release → concept + version DOIs recorded (step 4)
- [ ] DOI backfill PR (`CITATION.cff` `doi`/`identifiers`; README badge if the
      concept DOI changed)
- [ ] Release notes refreshed if needed

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
- **Zenodo mints the DOI via its REST API, not a GitHub-Release webhook.** We
  drive it with [`scripts/zenodo_release.py`](../scripts/zenodo_release.py)
  (token at `~/.zenodo_token`, production `zenodo.org`). **Publishing is
  permanent** — a published record can't be deleted, only superseded.
- Concept DOI **`10.5281/zenodo.20680195`** resolves to the latest version and
  never changes (it's the badge); each version gets its own version DOI
  (v0.0.1 = `…196`). The **release badge uses `?include_prereleases`** because
  0.0.x are GitHub *pre-releases*.
- The **codecov badge needs the `CODECOV_TOKEN`** repo secret (already set).

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

### 4. Mint the Zenodo DOI

```bash
python scripts/zenodo_release.py vX.Y.Z          # add --no-publish to review first
```

It fetches tags, `git archive`s the tag into a source tarball, creates the
deposition, uploads, attaches metadata, and **publishes** (permanent), printing
the **concept** and **version** DOIs. Record both. (Sanity precheck the token:
`curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $(cat ~/.zenodo_token)" https://zenodo.org/api/deposit/depositions?size=1` → `200`.)

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
- **Zenodo publish is irreversible.** Verify metadata (run with `--no-publish`
  and inspect the draft) before publishing. Sandbox (`--sandbox`) needs a
  *separate* token from `sandbox.zenodo.org`.
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
- [ ] `gh release create vX.Y.Z --prerelease` (tag + Release)
- [ ] `python scripts/zenodo_release.py vX.Y.Z` → concept + version DOIs recorded
- [ ] DOI backfill PR (`CITATION.cff` `doi`/`identifiers`; README badge if first
      release)
- [ ] Release notes refreshed if needed

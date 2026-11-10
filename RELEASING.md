# Releasing Paralingo

Each GitHub release publishes the package to PyPI and gets a permanent DOI from Zenodo.

## One-time setup

### 1. Zenodo (DOI) — do this before the first release

1. Sign in at https://zenodo.org with your GitHub account.
2. Open **GitHub** in the account menu, find `namashworks/paralingo` and switch it **On**.

Zenodo cannot be attached to a release that already exists, so turn it on first. It reads the authors,
title and licence from `CITATION.cff`.

### 2. PyPI Trusted Publishing (no API tokens)

1. Create accounts on https://pypi.org and https://test.pypi.org, and turn on two-factor authentication.
2. On each site, go to **Your account → Publishing → Add a new pending publisher** and enter:
   * PyPI project name: `paralingo`
   * Owner: `namashworks`
   * Repository: `paralingo`
   * Workflow: `release.yml`
   * Environment: `pypi` (on TestPyPI: `testpypi`)
3. In the GitHub repo, go to **Settings → Environments** and create `pypi` and `testpypi`.
   Optionally require your own approval on `pypi` so nothing publishes without you.

## Each release

1. Update the version in `pyproject.toml`, `paralingo/__init__.py` and `CITATION.cff`
   (`version` and `date-released`), and move the "Unreleased" notes in `CHANGELOG.md` under the new version.
2. Commit and push to `main`, and wait for CI to pass.
3. Optional dry run: **Actions → Release → Run workflow** uploads to TestPyPI. Then check it installs:
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple paralingo`
4. On GitHub: **Releases → Draft a new release**, tag `v1.0.0`, title `v1.0.0`, paste the changelog
   notes, and **Publish**. The workflow uploads to PyPI; Zenodo mints the DOI within a few minutes.

## After the first release

1. Copy the **concept DOI** from the Zenodo record. It always points to the latest version.
2. Add it to `CITATION.cff` as `doi: 10.5281/zenodo.XXXXXXX`.
3. Add the DOI badge Zenodo shows you to the top of `README.md`.

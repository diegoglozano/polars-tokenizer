# Releasing to PyPI

Releases publish prebuilt CPython 3.10–3.14 wheels for Linux x86-64 and arm64,
macOS Apple Silicon and Intel, and Windows x86-64, plus a source distribution.
The [release workflow](https://github.com/diegoglozano/polars-tokenizer/blob/main/.github/workflows/release.yml)
builds distributions on the five corresponding GitHub runners.

## First-release setup

The first publication creates the `polars-tokenizer` project on PyPI. In the
PyPI account that should own it, open **Publishing** and add a **pending trusted
publisher** with these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `polars-tokenizer` |
| GitHub owner | `diegoglozano` |
| Repository | `polars-tokenizer` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

The GitHub repository has a `pypi` environment restricted to `v*` tags.
Configure required reviewers there if releases should need an explicit approval. Trusted
publishing uses a short-lived GitHub OIDC identity; do not add a long-lived
PyPI token to repository secrets. A pending publisher does **not** reserve the
project name until the first upload succeeds.

## Preflight

1. Update both `pyproject.toml` and `Cargo.toml` to the same new version.
   Refresh `Cargo.lock` and `uv.lock` if needed. PyPI files and versions cannot
   be overwritten, so verify the version has not already been published.
2. Run the normal CI checks and build a local wheel and source distribution:

    ```bash
    uv sync --group dev --no-install-project
    uv run --no-sync maturin build --release --locked --compatibility pypi --out dist
    uv run --no-sync maturin sdist --out dist
    ```

3. Merge the release changes to `main`. Use **Run workflow** on the GitHub
   **Release** workflow to build the cross-platform artifacts without uploading
   anything. Check that all wheel, source-distribution, and wheel smoke-test
   jobs succeed.
4. Create and publish a GitHub Release tagged `vX.Y.Z` on the verified `main`
   commit. The tag must match the Python and Rust package versions. Publishing
   the GitHub Release triggers the same builds, then the isolated `pypi`
   environment job uploads the artifacts. Prereleases do not upload.
5. Check the PyPI file list, install a published wheel in a clean environment,
   and run a small token-count expression before announcing the release.

The workflow rejects mismatched Python/Rust versions, a release tag that does
not match the package version, and release commits not reachable from `main`.
The publish job has OIDC permission but does not check out source or build code.

For PyPI's account-side setup and security details, see the
[pending publisher guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

# Releasing XTC

XTC uses Git tags as the source of package versions. Do not edit a version number in
`pyproject.toml`: `setuptools-scm` derives it from the repository history.

## Development packages

Every successful push to `main` builds and publishes a development package to
TestPyPI. For example, commits after `xtc-v0.3.0` receive versions such as
`0.3.1.dev1` and `0.3.1.dev2`.

Install a development package while resolving dependencies from PyPI with:

```bash
python -m pip install \
  --index-url https://pypi.org/simple \
  --extra-index-url https://test.pypi.org/simple \
  "xtc-tools==0.3.1.dev1"
```

## Production release

Release tags must have the exact form `xtc-vX.Y.Z` and must point to a commit on
`main`. To publish a release:

```bash
git checkout main
git pull --ff-only upstream main
make check

git tag -a xtc-v0.3.0 -m "XTC 0.3.0"
git push upstream xtc-v0.3.0
```

The tag workflow checks that the package version is exactly `X.Y.Z`, builds and
validates the wheel and source distribution, and publishes them to PyPI. The
GitHub `pypi` environment should require maintainer approval.

Release tags and published versions are immutable. Never move or reuse a release
tag; create a new patch release instead.

## Trusted publishing setup

Publication uses PyPI trusted publishing and does not require an API token. The
repository administrators must configure two GitHub environments and matching
trusted publishers:

- `testpypi` for the `xtc-tools` project on TestPyPI;
- `pypi` for the `xtc-tools` project on PyPI.

For each trusted publisher, use:

- GitHub owner: `xtc-tools`;
- repository: `xtc`;
- workflow: `xtc-tests.yml`;
- environment: `testpypi` or `pypi`, respectively.

Protect the `xtc-v*` tag pattern so that only release maintainers can create or
delete production release tags.

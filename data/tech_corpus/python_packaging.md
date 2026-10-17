# Python Packaging — Technical Reference

## Overview

Python packages are distributed units of code managed by the Python Package Index (PyPI).
Modern Python projects declare their metadata and dependencies in `pyproject.toml`, introduced
by PEP 517 and PEP 518. The build backend (such as Hatchling, Setuptools, or Flit) reads
`pyproject.toml` and produces distribution archives.

## pyproject.toml Structure

A minimal `pyproject.toml` has three sections:

```
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-package"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = ["fastapi>=0.100", "httpx>=0.25"]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.1"]
```

### Dependency Specifiers

Dependencies follow PEP 440. Common operators:

- `==1.2.3` — exact pin
- `>=1.2,<2` — compatible range
- `~=1.4.2` — compatible release (equivalent to `>=1.4.2,<1.5`)
- `[extra]` — optional dependency group, e.g., `httpx[http2]`

### Version Numbering

Python packages use Semantic Versioning (`MAJOR.MINOR.PATCH`):
- MAJOR: breaking API changes
- MINOR: backward-compatible new features
- PATCH: backward-compatible bug fixes

Pre-release suffixes: `.a1` (alpha), `.b1` (beta), `.rc1` (release candidate).

## Dependency Management Tools

### pip

`pip` is the standard Python package installer. Key commands:

```bash
pip install package-name
pip install "package-name>=1.0,<2"
pip install -e .          # editable install (development)
pip install -r requirements.txt
pip freeze > requirements.txt
pip list --outdated
pip show package-name
```

### uv

`uv` is a fast Python package installer and resolver written in Rust. It is
compatible with `pip` and `pyproject.toml`. Key commands:

```bash
uv pip install package-name
uv pip install -e ".[dev]"
uv venv --python 3.12 .venv
uv pip sync requirements.txt
```

`uv` REPLACES `pip` in projects that adopt it. It resolves dependencies up to
100x faster than `pip` because it uses a SAT solver implemented in Rust.

### pip-tools

`pip-tools` provides `pip-compile` to generate pinned `requirements.txt` from
`pyproject.toml` and `pip-sync` to install exactly those pins.

## Virtual Environments

A virtual environment isolates Python packages per project:

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows PowerShell
source .venv/bin/activate  # Linux / macOS
```

The activated environment's `pip` and `python` are project-local.

## Package Discovery

`hatchling` discovers packages automatically. Override with:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/my_package"]
```

`src` layout keeps the package out of the root so accidental imports from the
project root are not possible during testing.

## Entry Points and Scripts

Declare CLI scripts in `pyproject.toml`:

```toml
[project.scripts]
my-cli = "my_package.cli:main"
```

After installation, `my-cli` is available on `PATH`.

## Publishing

```bash
python -m build          # creates dist/*.whl and dist/*.tar.gz
twine upload dist/*      # publishes to PyPI
```

Requires a PyPI account and API token configured in `~/.pypirc`.

## Dependency Conflicts

When two packages require incompatible versions of a shared dependency, the
resolver raises a `ResolutionImpossible` error. Resolution strategies:

1. Pin to an older compatible version of one package.
2. Use a virtual environment per project.
3. Check whether a newer release resolves the conflict.
4. Use `pip install --upgrade` to let the resolver attempt an upgrade path.

## Extras and Conditional Dependencies

```toml
[project.optional-dependencies]
http2 = ["h2>=4"]
all = ["my-package[http2]"]
```

Install with: `pip install "my-package[http2]"`

## License and Metadata

```toml
[project]
license = { text = "MIT" }
authors = [{ name = "Author Name", email = "author@example.com" }]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
]
```

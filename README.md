# PERCUSION Overview Repository

This repository contains analysis and visualization scripts for the overview paper of the **PERCUSION** campaign (*Persistent EarthCARE Underflight Studies of the ITCZ and Organized Convection*).

PERCUSION is one of eight sub-campaigns of [**ORCESTRA**](https://orcestra-campaign.org/intro.html) (*Organized Convection and EarthCARE Studies over the Tropical Atlantic*), conducted over the tropical Atlantic in August–September 2024.

---

## Repository Structure

### Python Package: `percusion`

The repository includes the Python package **`percusion`**, which provides core functionality for data analysis and plotting.

---

## Installation

To install the package in **editable (development) mode**, run:

```bash
pip install -e .
```

Run this command from the **root directory of the repository**.

After installation, the package can be imported in Python:

``` python
import percusion
```

Editable mode ensures that any changes to the source code are
immediately available without reinstalling the package.

## ShareLaTeX integration

This repository is linked to a ShareLaTeX project to facilitate collaborative editing.

## Getting started

Clone the repository and navigate into it:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

You can find the ShareLaTeX Git URL in the project’s Git settings.

### Synchronisation

Pull changes from ShareLaTeX:

```bash
git pull sharelatex master
```

Push local changes to ShareLaTeX:

```bash
git push sharelatex main:master
```

### Notes

- The local `main` branch is mapped to ShareLaTeX’s `master` branch.
- Always pull before pushing to avoid conflicts with edits made directly in ShareLaTeX.
- Resolve any merge conflicts locally before pushing changes back.

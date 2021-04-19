# Pre-commit all the way down

pre-commit hooks to run [`pre-commit`](https://pypi.org/project/pre-commit/) on
code snippets embeded in files that don't match the language used.

The primary (and currently only) application for this is to apply
Python-targeted hooks to Python snippets included in documentation files.

This project is inspired by the great
[`blacken-docs`](https://pypi.org/project/blacken-docs/).

## Install

The package is not available on Pypi yet. In the meantime it can be installed as
```shell
pip install git+https://github.com/cphyc/pre-commit-all-the-way-down.git#egg=pre_commit_all_the_way_down
```
or
```shell
git clone https://github.com/cphyc/pre-commit-all-the-way-down.git
cd pre-commit-all-the-way-down
pip install .
```

## Usage

### python-doc
This hook applies pre-commit hooks to Python blocks in `.rst` files.
It does so in 4 steps
1. extract Python code snippets
2. write their content to temporary files
3. run `pre-commit` hooks against these files
4. write back the to the original file.

The executable has a single option, `--whitelist` which allows to explicitly list which `pre-commit` hooks will be run.
For example
```shell
# will apply all pre-commit hooks to the file
python-doc documentation.rst

# will only apply black & isort
python-doc documentation.rst --whitelist black --whitelist isort
```

## Usage with pre-commit

The package has been tailored to fit in the `pre-commit` machinery. To use it, add the following to your `.pre-commit-config.yaml` file
```yaml
-   repo: https://github.com/cphyc/pre-commit-all-the-way-down
    rev: v0.0.2
    hooks:
    -   id: python-doc
```

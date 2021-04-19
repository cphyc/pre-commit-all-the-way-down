# Pre-commit all the way down

Run [`pre-commit`](https://pypi.org/project/pre-commit/) on python code blocks
in documentation file. This is inspired from the great
[`blacken-docs`](https://pypi.org/project/blacken-docs/).

## Install

The package is not on Pypi yet, but in the meantime you can install it via
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

This provides a single executable, `python-doc` which will do the following:
1. extract Python code snippets from `rst` files,
2. write their content to temporary files,
3. apply them `pre-commit` on these files using the configuration from the current directory,
4. write back the eventually-modified content into the original file.

The executable has a single option, `--whitelist` which allows to explicitly list which `pre-commit` hooks will be run.
For example
```shell
python-doc documentation.rst  # will apply all pre-commit hooks to the file
python-doc documentation.rst --whitelist black --whitelist isort  # will only apply black & isort
```

## Usage with pre-commit

The package has been tailored to fit in the `pre-commit` machinery. To use it, add the following to your `.pre-commit-config.yaml` file
```yaml
-   repo: https://github.com/cphyc/pre-commit-all-the-way-down
    rev: v0.0.1
    hooks:
    -   id: python-doc
```

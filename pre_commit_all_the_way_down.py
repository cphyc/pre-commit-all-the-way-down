import argparse
import re
from pathlib import Path
from re import Match
from sys import exit
from tempfile import TemporaryDirectory
from textwrap import dedent, indent
from typing import Optional, Sequence

import sh

BLOCK_TYPES = "(code|code-block|sourcecode|ipython)"
PY_LANGS = "(python|py|sage|python3|py3|numpy)"
DOCTEST_TYPES = "(testsetup|testcleanup|testcode)"
RST_RE = re.compile(
    rf"(?P<before>"
    rf"^(?P<indent> *)\.\. ("
    rf"jupyter-execute::|"
    rf"python-script::|"
    rf"{BLOCK_TYPES}:: {PY_LANGS}|"
    rf"{DOCTEST_TYPES}::.*"
    rf")\n"
    rf"((?P=indent) +:.*\n)*"
    rf"\n*"
    rf")"
    rf"(?P<code>(^((?P=indent) +.*)?\n)+)",
    re.MULTILINE,
)
INDENT_RE = re.compile("^ +(?=[^ ])", re.MULTILINE)
TRAILING_NL_RE = re.compile(r"\n+\Z", re.MULTILINE)


pre_commit = sh.Command("pre-commit").bake("run", "--files")


def fix_block(block: str, filename: str):
    """
    Fixes a code block.

    Parameters
    ----------
    block : str
        The block to fix
    filename : str
        The name of the file it originates from (used for logging)
    lineno : int
        The linenumber in `filename` where the block starts.
    """

    with TemporaryDirectory(dir=".") as d:
        fname = Path(d) / "script.py"
        with fname.open("w") as f:
            f.write(block)
        try:
            pre_commit(str(fname))
        except sh.ErrorReturnCode as e:
            # FIXME: do something with error, e.stdout
            print(e.stdout.decode())
        with fname.open("r") as f:
            newBlock = f.read()

    return newBlock


def fmt_source(src: str, fname: str) -> str:
    def _rst_match(match: Match[str]) -> str:
        # From https://github.com/asottile/blacken-docs/blob/ef58f2fcf2edbea87ba13d7463c13a9e7b282c1c/blacken_docs.py#L99-L99  # noqa
        min_indent = min(INDENT_RE.findall(match["code"]))
        trailing_ws_match = TRAILING_NL_RE.search(match["code"])
        assert trailing_ws_match
        trailing_ws = trailing_ws_match.group()
        code = TRAILING_NL_RE.sub("", match["code"])
        code = dedent(code)
        # Add a trailing new line to prevent pre-commit to fail because of that
        code += "\n"
        code = fix_block(code, fname)
        code = indent(code, min_indent)
        return f'{match["before"]}{code.rstrip()}{trailing_ws}'

    src = RST_RE.sub(_rst_match, src)
    return src


def format_file(filename: str) -> int:
    with open(filename, encoding="UTF-8") as f:
        contents = f.read()

    newContents = fmt_source(contents, filename)
    if newContents != contents:
        print(f"{filename}: Rewriting...")
        with open(filename, mode="w") as f:
            f.write(newContents)
        return 1
    else:
        return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    retv = 0
    for filename in args.filenames:
        retv |= format_file(filename)

    return retv


if __name__ == "__main__":
    exit(main())

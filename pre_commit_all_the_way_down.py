from glob import glob
from textwrap import dedent, indent
from typing import List

from tempfile import TemporaryDirectory
from pathlib import Path
from difflib import context_diff

import sh

pre_commit = sh.Command("pre-commit").bake("run", "--files")

def fix_block(block: str, filename: str, lineno: int):
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

    block = dedent(block)
    # FIXME: this should be caught upstream
    # remove head and trailing spaces
    while block[0] == "\n":
        block = block[1:]
    while block[-1] == "\n":
        block = block[:-1]
    block += "\n"

    with TemporaryDirectory(dir=".") as d:
        fname = Path(d) / "script.py"
        with fname.open("w") as f:
            f.write(block)
        try:
            out = pre_commit(str(fname))
        except sh.ErrorReturnCode as e:
            print(f"In {filename}:{lineno}")
            print(indent(e.stdout.decode(), prefix="   "))
        with fname.open("r") as f:
            newBlock = f.read()

    return indent(
        newBlock,
        prefix="   ",
    )


def fix_python_docstrings(lines: List[str], fname: str):
    """
    Fixes docstrings in .rst files.

    Parameters
    ----------
    lines : List[str]
        A list of lines. See notes for what is fixed.
    fname : str
        The name of the file (used for logging)

    Returns
    -------
    The modified content as a list of lines.

    Notes
    -----
    The fixed blocks have the form
    >>> .. python-script::
    ...
    ...    import numpy as np
    ...    [...]

    """
    state = 0
    newLines = []
    for iline, line in enumerate(lines):
        line = line.rstrip()
        # Start of block
        if line == ".. python-script::":
            state = 1
            start = iline + 1
            newLines.append(line)
        elif state == 1 and not (line.startswith(" ") or line == ""):
            block = "".join(lines[start:iline])
            newLines.append("")
            newLines.extend(fix_block(block, fname, start).split("\n"))
            newLines.append(line)
            state = 0
        elif state == 0:
            newLines.append(line)

    # Happens if we reach EOF
    if state == 1:
        block = "".join(lines[start:iline])
        newLines.append("")
        newLines.extend(fix_block(block, fname, start).split("\n"))
        newLines.append(line)

    return newLines

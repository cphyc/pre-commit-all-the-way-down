import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from re import Match
from tempfile import TemporaryDirectory
from textwrap import dedent, indent
from typing import Callable, Optional, Sequence

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
REPL_RE = re.compile(
    "(?P<content>(?P<indent> *)>{3} .*\n((?P=indent)\\.{3} .*\n)*)", re.MULTILINE
)
INDENT_RE = re.compile("^ +(?=[^ ])", re.MULTILINE)
TRAILING_NL_RE = re.compile(r"\n+\Z", re.MULTILINE)


def apply_pre_commit_on_block(block: str, whitelist: Sequence[Optional[str]]) -> str:
    """
    Fix a code block.

    Parameters
    ----------
    block : str
        The block to fix

    Notes
    -----
    The code block should have no leading indentation and be valid
    Python.
    """

    def _pre_commit_helper(fname: str, hook_id: Optional[str]):
        args = ["pre-commit", "run"]
        if hook_id is not None:
            args.append(hook_id)
        args.extend(("--files", fname))
        try:
            subprocess.run(args, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(e.stderr.decode(), file=sys.stderr)
            print(e.stdout.decode())

    if not whitelist:
        whitelist = [None]

    with TemporaryDirectory() as d:
        fname = Path(d) / "script.py"
        fname.write_text(block)
        for wl in whitelist:
            _pre_commit_helper(str(fname), wl)

        newBlock = fname.read_text()
    return newBlock


def walk_ast_helper(callback: Callable[[Match[str]], str], src: str) -> str:
    lines = src.splitlines()
    newLines = lines.copy()

    # Extract all functions and classes
    tree = ast.parse(src)
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module))
    ]

    # Iterate over docstrings in reversed order so that lines
    # can be modified
    for node in reversed(nodes):
        docstring = ast.get_docstring(node)
        if not docstring:
            continue

        doc_node = node.body[0]
        docstring_lines = lines[doc_node.lineno - 1 : doc_node.end_lineno]
        doc = "\n".join(docstring_lines)

        doc = REPL_RE.sub(callback, doc)
        newLines = (
            newLines[: doc_node.lineno - 1]
            + doc.splitlines()
            + newLines[doc_node.end_lineno :]
        )
    return "\n".join(newLines)


def apply_pre_commit_on_str(src: str, fname: str, whitelist: Sequence[str]) -> str:
    # The _*_match functions are adapted from
    # https://github.com/asottile/blacken-docs
    def _rst_match(match: Match[str]) -> str:
        min_indent = min(INDENT_RE.findall(match["code"]))
        trailing_ws_match = TRAILING_NL_RE.search(match["code"])
        assert trailing_ws_match
        trailing_ws = trailing_ws_match.group()
        code = TRAILING_NL_RE.sub("", match["code"])
        code = dedent(code)
        # Add a trailing new line to prevent pre-commit to fail because of that
        code += "\n"
        code = apply_pre_commit_on_block(code, whitelist)
        code = indent(code, min_indent)
        return f'{match["before"]}{code.rstrip()}{trailing_ws}'

    def _pycon_match(match: Match[str]) -> str:
        head_ws = match["indent"]
        block = "\n".join(
            line[len(head_ws) + 4 :] for line in match["content"].splitlines()
        )
        tmp_code = apply_pre_commit_on_block(block, whitelist)
        code_lines = []
        for i, line in enumerate(tmp_code.splitlines()):
            if i == 0:
                code_lines.append(f"{head_ws}>>> {line}\n")
            else:
                code_lines.append(f"{head_ws}... {line}\n")
        return "".join(code_lines)

    # src = RST_RE.sub(_rst_match, src)
    src = walk_ast_helper(_pycon_match, src)

    return src


def apply_pre_commit_on_file(filename: str, whitelist: Sequence[str]) -> int:
    with open(filename, encoding="UTF-8") as f:
        contents = f.read()

    newContents = apply_pre_commit_on_str(contents, filename, whitelist)
    if newContents != contents:
        print(f"Rewriting {filename}", file=sys.stderr)
        with open(filename, mode="w") as f:
            f.write(newContents)
        return 1

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    parser.add_argument(
        "--whitelist",
        nargs="+",
        default=[],
        type=str,
        help="A whitelist of hook ids to run",
    )
    args = parser.parse_args(argv)

    retv = 0
    for filename in args.filenames:
        retv += apply_pre_commit_on_file(filename, args.whitelist)

    return retv


if __name__ == "__main__":
    sys.exit(main())

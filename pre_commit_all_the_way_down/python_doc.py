import argparse
import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent, indent
from typing import Callable, List, Match, Optional, Sequence, Tuple

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


@dataclass
class Context:
    filename: str
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class Error:
    context: Context
    exception: Exception
    stdout: str
    stderr: str


def apply_pre_commit_on_block(
    block: str,
    context: Context,
    whitelist: Optional[Sequence[str]] = None,
    skiplist: Optional[Sequence[str]] = None,
) -> Tuple[int, str, List[Error]]:
    """
    Fix a code block.

    Parameters
    ----------
    block : str
        The block to fix
    context : Context
        The context of this block
    whitelist, skiplist : Sequence[str]
        Which hooks to whitelist/skip.

    Notes
    -----
    The code block should have no leading indentation and be valid
    Python.
    """
    ret = 0
    errors = []
    env = os.environ.copy()
    if skiplist:
        env["SKIP"] = ",".join(_ for _ in skiplist if _)

    def _pre_commit_helper(fname: str, hook_id: Optional[str]):
        nonlocal ret, errors
        args = ["pre-commit", "run"]
        if hook_id is not None:
            args.append(hook_id)
        args.extend(("--files", fname))
        try:
            state = subprocess.run(args, check=True, capture_output=True, env=env)
            ret |= state.returncode
        except subprocess.CalledProcessError as e:
            errors.append(
                Error(
                    context=context,
                    exception=e,
                    stdout=e.stdout.decode(),
                    stderr=e.stderr.decode(),
                )
            )
            ret |= e.returncode

    with TemporaryDirectory() as d:
        fname = Path(d) / "script.py"
        fname.write_text(block)
        if whitelist:
            for wl in whitelist:
                _pre_commit_helper(str(fname), wl)
        else:
            _pre_commit_helper(str(fname), None)

        newBlock = fname.read_text()

    return ret, newBlock, errors


def walk_ast_helper(
    callback: Callable[[Match[str], Context, str], str], src: str, context: Context
) -> str:
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
    nodes_by_decr_lineno = sorted(
        nodes, key=lambda node: node.body[0].lineno if node.body else 0, reverse=True
    )
    for node in nodes_by_decr_lineno:
        docstring = ast.get_docstring(node)
        if not docstring:
            continue

        doc_node = node.body[0]
        docstring_lines = lines[doc_node.lineno - 1 : doc_node.end_lineno]
        newContext = Context(
            filename=context.filename, start=doc_node.lineno, end=doc_node.end_lineno
        )
        doc = "\n".join(docstring_lines)
        doc = REPL_RE.sub(partial(callback, context=newContext, docStringSrc=doc), doc)

        newLines = (
            newLines[: doc_node.lineno - 1]
            + doc.splitlines()
            + newLines[doc_node.end_lineno :]
        )

    formatted = "\n".join(newLines) + "\n"
    # Return an empty file if it is empty
    if not formatted.strip():
        return ""
    return formatted


def fake_indent(block: str, level: int) -> str:
    i = 0
    newBlock = ""
    while i * 4 < level:
        newBlock += "    " * i + "if True:\n"
        i += 1

    newBlock += indent(block, " " * (i * 4))
    return newBlock


def fake_dedent(block: str, level: int) -> str:
    i = 0
    lines = block.splitlines()
    while i * 4 < level:
        i += 1
    return dedent("\n".join(lines[i:]))


def offset_to_lineno(src: str, offset: int) -> int:
    return src[:offset].count("\n")


def apply_pre_commit_rst(
    src: str,
    context: Context,
    *,
    whitelist: Optional[Sequence[str]] = None,
    skiplist: Optional[Sequence[str]] = None,
) -> Tuple[int, str, List[Error]]:
    errors: List[Error] = []
    ret = 0

    # The _*_match functions are adapted from
    # https://github.com/asottile/blacken-docs
    def _rst_match(match: Match[str]) -> str:
        nonlocal ret, errors
        min_indent = min(INDENT_RE.findall(match["code"]))
        trailing_ws_match = TRAILING_NL_RE.search(match["code"])
        assert trailing_ws_match
        trailing_ws = trailing_ws_match.group()
        code = TRAILING_NL_RE.sub("", match["code"])
        code = dedent(code)
        code = fake_indent(code, len(min_indent))
        # Add a trailing new line to prevent pre-commit to fail because of that
        code += "\n"
        newContext = Context(
            filename=context.filename,
            start=offset_to_lineno(src, match.start()),
            end=offset_to_lineno(src, match.end()),
        )
        retcode, code, newErrors = apply_pre_commit_on_block(
            code, newContext, whitelist=whitelist, skiplist=skiplist
        )
        errors.extend(newErrors)

        ret |= retcode
        code = fake_dedent(code, len(min_indent))
        code = code.strip()
        code = indent(code, min_indent)
        return f'{match["before"]}{code}{trailing_ws}'

    src = RST_RE.sub(_rst_match, src)

    return ret, src, errors


def apply_pre_commit_pydocstring(
    src: str,
    context: Context,
    *,
    whitelist: Optional[Sequence[str]] = None,
    skiplist: Optional[Sequence[str]] = None,
) -> Tuple[int, str, List[Error]]:
    errors: List[Error] = []
    ret = 0

    def _pycon_match(match: Match[str], context: Context, docStringSrc: str) -> str:
        nonlocal ret, errors
        head_ws = match["indent"]
        trailing_ws_match = TRAILING_NL_RE.search(match["content"])
        assert trailing_ws_match
        trailing_ws = trailing_ws_match.group()
        code = "\n".join(
            line[len(head_ws) + 4 :] for line in match["content"].splitlines()
        )
        # Special case for
        # >>> # line with only a comment
        if code.count("\n") == 0 and code.startswith("#"):
            return match.group(0)
        code = fake_indent(code, len(head_ws) + 4)
        docStringStart = context.start if context.start is not None else 0
        start = docStringStart + offset_to_lineno(docStringSrc, match.start())
        end = docStringStart + offset_to_lineno(docStringSrc, match.end())

        newContext = Context(filename=context.filename, start=start, end=end)
        retcode, code, newErrors = apply_pre_commit_on_block(
            code, newContext, whitelist=whitelist, skiplist=skiplist
        )
        errors.extend(newErrors)

        ret |= retcode
        code = fake_dedent(code, len(head_ws) + 4)
        code = code.strip()
        code_lines = []
        for i, line in enumerate(code.splitlines()):
            # Skip empty lines
            if line.strip() == "":
                continue
            if i == 0:
                code_lines.append(f"{head_ws}>>> {line}\n")
            else:
                code_lines.append(f"{head_ws}... {line}\n")
        newContent = "".join(code_lines)
        return f"{newContent.rstrip()}{trailing_ws}"

    src = walk_ast_helper(_pycon_match, src, context)

    return ret, src, errors


def print_errors(errors: List[Error]):
    for error in errors:
        msg = []
        ctx = error.context
        msg.append(f"Error in {ctx.filename}:{ctx.start}:{ctx.end}")
        stdout = error.stdout.rstrip()
        if stdout.strip():
            msg.append("STDOUT was:")
            msg.append(indent(stdout, " | ", predicate=lambda _line: True))
        stderr = error.stderr.rstrip()
        if stderr.strip():
            msg.append("STDERR was:")
            msg.append(indent(stderr, " | ", predicate=lambda _line: True))

        print("\n".join(msg), file=sys.stderr)


def apply_pre_commit_on_file(
    filename: str,
    *,
    whitelist: Optional[Sequence[str]] = None,
    skiplist: Optional[Sequence[str]] = None,
    write_back=True,
) -> int:
    with open(filename, encoding="UTF-8") as f:
        contents = f.read()

    _filename, extension = os.path.splitext(filename)
    context = Context(filename=filename)
    if extension == ".rst":
        retcode, newContents, errors = apply_pre_commit_rst(
            contents, context, whitelist=whitelist, skiplist=skiplist
        )
    elif extension == ".py":
        retcode, newContents, errors = apply_pre_commit_pydocstring(
            contents, context, whitelist=whitelist, skiplist=skiplist
        )
    else:
        return 0

    print_errors(errors)

    if newContents != contents and write_back:
        print(f"Rewriting {filename}", file=sys.stderr)
        with open(filename, mode="w") as f:
            f.write(newContents)
        return 1 | retcode

    return retcode


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    group = parser.add_argument_group()
    group.add_argument(
        "--whitelist",
        action="append",
        default=[],
        type=str,
        help="A whitelist of hook ids to run",
    )
    group.add_argument(
        "--skiplist",
        action="append",
        default=["python-doc"],
        type=str,
        help="A skiplist of hook ids to skip",
    )
    parser.add_argument("--no-write-back", dest="write_back", action="store_false")
    args = parser.parse_args(argv)

    retv = 0
    for filename in args.filenames:
        retv += apply_pre_commit_on_file(
            filename,
            whitelist=args.whitelist,
            skiplist=args.skiplist,
            write_back=args.write_back,
        )

    return retv


if __name__ == "__main__":
    sys.exit(main())

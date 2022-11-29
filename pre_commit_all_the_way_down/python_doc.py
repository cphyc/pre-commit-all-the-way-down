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
from typing import Callable, Match, Sequence

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
    start: int | None = None
    end: int | None = None


@dataclass
class Error:
    context: Context
    exception: Exception
    stdout: str
    stderr: str


def apply_pre_commit_on_block(
    block: str,
    context: Context,
    whitelist: Sequence[str] | None = None,
    skiplist: Sequence[str] | None = None,
) -> tuple[int, str, list[Error]]:
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

    def _pre_commit_helper(fname: str, hook_id: str | None):
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

        new_block = fname.read_text()

    return ret, new_block, errors


def walk_ast_helper(
    callback: Callable[[Match[str], Context, str], str], src: str, context: Context
) -> str:
    lines = src.splitlines()
    new_lines = lines.copy()

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
        new_context = Context(
            filename=context.filename, start=doc_node.lineno, end=doc_node.end_lineno
        )
        doc = "\n".join(docstring_lines)
        doc = REPL_RE.sub(partial(callback, context=new_context, docStringSrc=doc), doc)

        new_lines = (
            new_lines[: doc_node.lineno - 1]
            + doc.splitlines()
            + new_lines[doc_node.end_lineno :]
        )

    formatted = "\n".join(new_lines) + "\n"
    # Return an empty file if it is empty
    if not formatted.strip():
        return ""
    return formatted


def fake_indent(block: str, level: int) -> str:
    i = 0
    new_block = ""
    while i * 4 < level:
        new_block += "    " * i + "if True:\n"
        i += 1

    new_block += indent(block, " " * (i * 4))
    return new_block


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
    whitelist: Sequence[str] | None = None,
    skiplist: Sequence[str] | None = None,
) -> tuple[int, str, list[Error]]:
    errors: list[Error] = []
    ret = 0

    # The _*_match functions are adapted from
    # https://github.com/asottile/blacken-docs
    def _rst_match(match: Match[str]) -> str:
        nonlocal ret, errors
        min_indent = min(INDENT_RE.findall(match["code"]))
        trailing_ws_match = TRAILING_NL_RE.search(match["code"])
        if not trailing_ws_match:
            raise RuntimeError("No trailing whitespace found")
        trailing_ws = trailing_ws_match.group()
        code = TRAILING_NL_RE.sub("", match["code"])
        code = dedent(code)
        code = fake_indent(code, len(min_indent))
        # Add a trailing new line to prevent pre-commit to fail because of that
        code += "\n"
        new_context = Context(
            filename=context.filename,
            start=offset_to_lineno(src, match.start()),
            end=offset_to_lineno(src, match.end()),
        )
        retcode, code, new_errors = apply_pre_commit_on_block(
            code, new_context, whitelist=whitelist, skiplist=skiplist
        )
        errors.extend(new_errors)

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
    whitelist: Sequence[str] | None = None,
    skiplist: Sequence[str] | None = None,
) -> tuple[int, str, list[Error]]:
    errors: list[Error] = []
    ret = 0

    def _pycon_match(match: Match[str], context: Context, doc_string_src: str) -> str:
        nonlocal ret, errors
        head_ws = match["indent"]
        trailing_ws_match = TRAILING_NL_RE.search(match["content"])
        if not trailing_ws_match:
            raise RuntimeError("No trailing whitespace found")
        trailing_ws = trailing_ws_match.group()
        code = "\n".join(
            line[len(head_ws) + 4 :] for line in match["content"].splitlines()
        )
        # Special case for
        # >>> # line with only a comment
        if code.count("\n") == 0 and code.startswith("#"):
            return match.group(0)
        code = fake_indent(code, len(head_ws) + 4)
        doc_string_start = context.start if context.start is not None else 0
        start = doc_string_start + offset_to_lineno(doc_string_src, match.start())
        end = doc_string_start + offset_to_lineno(doc_string_src, match.end())

        new_context = Context(filename=context.filename, start=start, end=end)
        retcode, code, new_errors = apply_pre_commit_on_block(
            code, new_context, whitelist=whitelist, skiplist=skiplist
        )
        errors.extend(new_errors)

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
        new_content = "".join(code_lines)
        return f"{new_content.rstrip()}{trailing_ws}"

    src = walk_ast_helper(_pycon_match, src, context)

    return ret, src, errors


def print_errors(errors: list[Error]):
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
    whitelist: Sequence[str] | None = None,
    skiplist: Sequence[str] | None = None,
    write_back=True,
) -> int:
    with open(filename, encoding="UTF-8") as f:
        contents = f.read()

    _filename, extension = os.path.splitext(filename)
    context = Context(filename=filename)
    if extension == ".rst":
        retcode, new_contents, errors = apply_pre_commit_rst(
            contents, context, whitelist=whitelist, skiplist=skiplist
        )
    elif extension == ".py":
        retcode, new_contents, errors = apply_pre_commit_pydocstring(
            contents, context, whitelist=whitelist, skiplist=skiplist
        )
    else:
        return 0

    print_errors(errors)

    if new_contents != contents and write_back:
        print(f"Rewriting {filename}", file=sys.stderr)
        with open(filename, mode="w") as f:
            f.write(new_contents)
        return 1 | retcode

    return retcode


def main(argv: Sequence[str] | None = None) -> int:
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
    sys.exit(main(sys.argv[1:]))

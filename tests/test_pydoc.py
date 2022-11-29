from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

from pre_commit_all_the_way_down.python_doc import (
    Context,
    apply_pre_commit_on_file,
    apply_pre_commit_pydocstring,
)

before = (
    '''\
def foo():
    """
    >>> import numpy as np
    >>> from foo import c, b, a
    >>> y=1+1+(2
    ... +4)
    >>> x = "This is a very long chunk that should be split"'''
    + ''' "around" "here" "to span" "multiple" "lines"  # noqa: E501
    """
    pass
'''
)

after = dedent(
    '''\
    def foo():
        """
        >>> import numpy as np
        >>> from foo import a, b, c
        >>> y = 1 + 1 + (2 + 4)
        >>> x = (
        ...     "This is a very long chunk that should be split"
        ...     "around"
        ...     "here"
        ...     "to span"
        ...     "multiple"
        ...     "lines"
        ... )  # noqa: E501
        """
        pass
    '''
)


def test_format_from_string():
    context = Context(filename="dummy")

    (*out, _errors) = apply_pre_commit_pydocstring(
        before, context=context, skiplist=["end-of-file-fixer", "flake8"]
    )
    assert out == [0, after]

    (*out, _errors) = apply_pre_commit_pydocstring(
        before, context=context, whitelist=["black", "isort"]
    )
    assert out == [0, after]

    (*out, _errors) = apply_pre_commit_pydocstring(
        before, context=context, whitelist=["black", "isort", "flake8"]
    )
    assert out == [1, after]


def test_format_from_file():
    with TemporaryDirectory() as dirname:
        f = Path(dirname) / "test.py"

        f.write_text(before)
        retcode = apply_pre_commit_on_file(
            str(f), skiplist=["flake8", "end-of-file-fixer"]
        )
        modified = f.read_text()
        assert (retcode, modified) == (1, after)

        f.write_text(before)
        retcode = apply_pre_commit_on_file(
            str(f), whitelist=["black", "isort", "flake8"]
        )
        modified = f.read_text()
        assert (retcode, modified) == (1, after)

        f.write_text(before)
        retcode = apply_pre_commit_on_file(str(f), whitelist=["black", "isort"])
        modified = f.read_text()
        assert (retcode, modified) == (1, after)

        # NOTE: not updating the content this time, it should be unchanged now!
        retcode = apply_pre_commit_on_file(str(f), whitelist=["black", "isort"])
        modified = f.read_text()
        assert (retcode, modified) == (0, after)

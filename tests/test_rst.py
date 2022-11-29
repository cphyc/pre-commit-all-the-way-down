from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

from pre_commit_all_the_way_down.python_doc import (
    Context,
    apply_pre_commit_on_file,
    apply_pre_commit_rst,
)


def test_format_from_string():
    before = dedent(
        """\
        .. python-script::

            import toto
            import numpy as np
            np.random.rand(10,10,10)
            msg = 'yeeeeeet';
        """
    )
    after = dedent(
        """\
        .. python-script::

            import numpy as np
            import toto

            np.random.rand(10, 10, 10)
            msg = "yeeeeeet"
        """
    )

    context = Context(filename="dummy")
    (*out, _errors) = apply_pre_commit_rst(before, context=context, skiplist=["flake8"])
    assert out == [1, after]

    (*out, _errors) = apply_pre_commit_rst(
        before, context=context, whitelist=["black", "isort"]
    )
    assert out == [0, after]

    (*out, _errors) = apply_pre_commit_rst(
        before, context=context, whitelist=["black", "isort", "flake8"]
    )
    assert out == [1, after]


def test_format_from_file():
    before = dedent(
        """\
        .. python-script::

            import toto
            import numpy as np
            np.random.rand(10,10,10)
            msg = 'yeeeeeet';
        """
    )

    after = dedent(
        """\
        .. python-script::

            import numpy as np
            import toto

            np.random.rand(10, 10, 10)
            msg = "yeeeeeet"
        """
    )

    with TemporaryDirectory() as dirname:
        f = Path(dirname) / "test.rst"

        f.write_text(before)
        retcode = apply_pre_commit_on_file(str(f), skiplist=["flake8"])
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

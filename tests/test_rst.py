from textwrap import dedent

from pre_commit_all_the_way_down.python_doc import apply_pre_commit_rst


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

    out = apply_pre_commit_rst(before, skiplist=["flake8"])
    assert out == (0, after)

    out = apply_pre_commit_rst(before, whitelist=["black", "isort"])
    assert out == (0, after)

    out = apply_pre_commit_rst(before, whitelist=["black", "isort", "flake8"])
    assert out == (1, after)

from textwrap import dedent

from pre_commit_all_the_way_down.python_doc import apply_pre_commit_on_str


def test_format_src_trivial():
    before = dedent(
        """\
        .. python-script::

            import toto
            import numpy as np
            np.random.rand(10,10,10)
        """
    )
    after = dedent(
        """\
        .. python-script::

            import numpy as np
            import toto

            np.random.rand(10, 10, 10)
        """
    )

    out = apply_pre_commit_on_str(before, [])
    assert out == after

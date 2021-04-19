from textwrap import dedent

from pre_commit_all_the_way_down.python_doc import apply_pre_commit_on_str


def test_format_python_script():
    before = dedent(
        """\
        .. python-script::

            import toto
            import numpy as np
            np.random.rand(10,10,10)
            msg = 'yeeeeeet';
            print('%s' % msg)
        """
    )
    after = dedent(
        """\
        .. python-script::

            import numpy as np
            import toto

            np.random.rand(10, 10, 10)
            msg = "yeeeeeet"
            print(f"{msg}")
        """
    )

    out = apply_pre_commit_on_str(before, [])
    assert out == after

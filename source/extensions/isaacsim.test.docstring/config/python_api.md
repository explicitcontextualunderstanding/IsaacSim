# Public API for module isaacsim.test.docstring:

## Classes

- class StandaloneDocTestCase(unittest.TestCase)
  - def __init__(self, *args, **kwargs)
  - def assertDocTest(self, expr: object, msg: str = '', flags: int = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS | doctest.FAIL_FAST)
  - def assertDocTests(self, expr: object, msg: str = '', flags: int = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS | doctest.FAIL_FAST, order: list[tuple[object, int]] = [], exclude: list[object] = [], stop_on_failure: bool = False)

- class AsyncDocTestCase(omni.kit.test.AsyncTestCase)
  - def __init__(self, *args, **kwargs)
  - def assertDocTest(self, expr: object, msg: str = '', flags: int = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS | doctest.FAIL_FAST)
  - async def assertDocTests(self, expr: object, msg: str = '', flags: int = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS | doctest.FAIL_FAST, order: list[tuple[object, int]] = [], exclude: list[object] = [], stop_on_failure: bool = False, await_update: bool = True)

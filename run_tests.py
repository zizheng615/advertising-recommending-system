"""零依赖测试运行器 —— pytest 装不上时的回退路径。

为什么要有这个
--------------
交付环境里 C 盘只剩 0.6GB，装任何包都可能失败。测试是「每个功能点实现后
必须测」这条要求的载体，不能因为缺一个测试框架就交不出去。

这里实现的是 pytest 的一个很小的子集，够用就行：

* 发现 ``tests/test_*.py``；
* 收集模块级的 ``test_*`` 函数和 ``Test*`` 类里的 ``test_*`` 方法；
* 解析函数签名的参数名，从 ``tests/conftest.py`` 的 fixture 表里取值；
* **按 fixture 名做会话级缓存** —— 数据生成、PCA、KMeans 各只跑一次，
  否则几十个测试会把同一份数据反复造几十遍。

行为上与 pytest 对齐的部分：``pytest.raises``、``pytest.mark.skip``、
``pytest.approx`` 这些常用工具照常可用（直接 import pytest 即可，
只要装了 pytest 就一定有；没装的话下面会退化）。

用法::

    python run_tests.py              # 跑全部
    python run_tests.py features     # 只跑文件名含 features 的
    python run_tests.py -v           # 打印每个用例名
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"

# 让 tests/ 里能直接 import config / src.*
sys.path.insert(0, str(ROOT))


class _Skipped(Exception):
    pass


def _install_pytest_shim() -> bool:
    """pytest 不可用时，塞一个最小替身进 sys.modules。

    为什么需要：``tests/conftest.py`` 用 ``@pytest.fixture`` 声明 fixture，
    真没装 pytest 时连 import 都过不去。塞一个只提供
    ``fixture / raises / approx / mark / skip`` 的替身，就能让同一份测试
    文件在两种环境下都跑得起来 —— 这正是「两种 runner 结果一致」的前提。

    Returns
    -------
    True 表示真的装上了替身（即环境里没有 pytest）。
    """
    try:
        import pytest  # noqa: F401
        return False
    except ImportError:
        pass

    import contextlib
    import types

    shim = types.ModuleType("pytest")

    def fixture(func=None, *, scope="function", **kw):
        def deco(f):
            f._is_fixture = True
            f._fixture_scope = scope
            return f
        return deco if func is None else deco(func)

    @contextlib.contextmanager
    def raises(exc, match=None):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"期望抛出 {exc.__name__}，但没有")

    def skip(reason=""):
        raise _Skipped(reason)

    class _Approx:
        def __init__(self, expected, rel=1e-6, abs=1e-12):
            self.expected, self.rel, self.abs = expected, rel, abs

        def __eq__(self, other):
            import numpy as np
            return bool(np.allclose(other, self.expected,
                                    rtol=self.rel, atol=self.abs))

    class _Mark:
        def __getattr__(self, name):
            def deco(*a, **k):
                if len(a) == 1 and callable(a[0]) and not k:
                    a[0].__rt_skip__ = True
                    return a[0]

                def inner(f):
                    if name in ("skip", "skipif"):
                        f.__rt_skip__ = True
                    return f
                return inner
            return deco

    shim.fixture = fixture
    shim.raises = raises
    shim.skip = skip
    shim.approx = _Approx
    shim.mark = _Mark()
    shim.Skipped = _Skipped
    shim.__version__ = "0.0.0-shim"
    sys.modules["pytest"] = shim
    return True


def _load_module(path: Path):
    """按文件路径加载模块，模块名带上前缀避免重名。"""
    name = f"_rt_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class FixtureRegistry:
    """fixture 缓存。

    ``conftest.py`` 里声明的 fixture 登记在**会话级**的 registry 里，
    测试文件自己声明的登记在**模块级**的 registry 里，
    后者通过 ``parent`` 链回前者。

    为什么要分两层（而不是全塞一个字典）：pytest 的语义就是这样 ——
    测试模块里的 fixture **遮蔽**同名的 conftest fixture，
    且模块级 fixture 的缓存不跨模块共享。混成一个字典时，
    某个模块定义的 ``cfg`` 会覆盖掉 conftest 的版本并被后面所有模块复用，
    两个 runner 的结果就不再一致了。
    """

    def __init__(self, parent: "FixtureRegistry | None" = None) -> None:
        self._parent = parent
        self._factories: dict[str, object] = {}
        self._cache: dict[str, object] = {}
        self._building: set[str] = set()

    @staticmethod
    def _unwrap_fixture(obj):
        """把 pytest 的 fixture 包装对象剥回原始函数。

        pytest 8 之前 ``@pytest.fixture`` 就是给函数挂一个
        ``_pytestfixturefunction`` 属性、原样返回；8 之后返回的是
        ``FixtureFunctionDefinition`` 包装对象，直接用签名解析会拿到
        ``(*args, **kwargs)``。两种形态都得认，否则 runner 会静默地
        取不到任何 fixture —— 所有用例一起报 KeyError。
        """
        fn = getattr(obj, "_get_wrapped_function", None)
        if callable(fn):
            return fn()
        return obj

    @classmethod
    def _is_fixture(cls, obj) -> bool:
        if not callable(obj):
            return False
        if getattr(obj, "_is_fixture", False):          # 我们的替身
            return True
        if hasattr(obj, "_pytestfixturefunction"):      # pytest < 8
            return True
        if type(obj).__name__ == "FixtureFunctionDefinition":   # pytest >= 8
            return True
        return False

    def register_all(self, mod) -> None:
        for name, obj in vars(mod).items():
            if self._is_fixture(obj):
                self._factories[name] = self._unwrap_fixture(obj)

    def get(self, name: str):
        if name in self._cache:
            return self._cache[name]
        if name not in self._factories:
            if self._parent is not None:
                return self._parent.get(name)
            raise KeyError(f"未定义的 fixture: {name}")
        if name in self._building:
            raise RuntimeError(f"fixture 循环依赖: {name}")
        self._building.add(name)
        try:
            fn = self._factories[name]
            kwargs = {p: self.get(p) for p in inspect.signature(fn).parameters}
            value = fn(**kwargs)
            self._cache[name] = value
            return value
        finally:
            self._building.discard(name)


def _cases(mod):
    """收集模块里的用例，返回 (显示名, 可调用对象) 列表。"""
    out = []
    for name, obj in vars(mod).items():
        if name.startswith("test") and inspect.isfunction(obj):
            out.append((name, obj))
        elif name.startswith("Test") and inspect.isclass(obj):
            inst = obj()
            for mname, meth in vars(obj).items():
                if mname.startswith("test") and callable(meth):
                    out.append((f"{name}::{mname}",
                                getattr(inst, mname)))
    return out


def _is_skipped(fn) -> bool:
    if getattr(fn, "__rt_skip__", False):
        return True
    for mark in getattr(fn, "pytestmark", []) or []:
        if getattr(mark, "name", "") in ("skip", "skipif"):
            return True
    return False


def _run_case(fn, registry: FixtureRegistry):
    """执行单个用例，返回 (状态, 信息)。状态 ∈ {pass, fail, skip}。"""
    if _is_skipped(fn):
        return "skip", ""
    params = list(inspect.signature(fn).parameters)
    try:
        kwargs = {p: registry.get(p) for p in params}
        fn(**kwargs)
        return "pass", ""
    except _Skipped as exc:
        return "skip", str(exc)
    except Exception as exc:                     # noqa: BLE001 — runner 必须兜住一切
        if type(exc).__name__ == "Skipped":
            return "skip", str(exc)
        return "fail", traceback.format_exc()


def main(argv: list[str]) -> int:
    verbose = "-v" in argv
    pattern = next((a for a in argv[1:] if not a.startswith("-")), "")

    shimmed = _install_pytest_shim()
    print(f"runner: {'零依赖回退模式（未装 pytest）' if shimmed else 'pytest 可用'}")

    if not TESTS.exists():
        print(f"找不到测试目录 {TESTS}")
        return 2

    files = sorted(p for p in TESTS.glob("test_*.py"))
    if pattern:
        files = [p for p in files if pattern in p.name]
    if not files:
        print(f"没有匹配 {pattern!r} 的测试文件")
        return 2

    base_registry = FixtureRegistry()
    conftest_path = TESTS / "conftest.py"
    if conftest_path.exists():
        base_registry.register_all(_load_module(conftest_path))

    total = passed = failed = skipped = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    for path in files:
        try:
            mod = _load_module(path)
        except Exception:                        # noqa: BLE001
            failed += 1
            total += 1
            failures.append((path.name, traceback.format_exc()))
            print(f"\n[导入失败] {path.name}")
            continue

        # 每个文件一个模块级 registry，回退到 conftest 的会话级 registry ——
        # 对应 pytest 里「模块 fixture 遮蔽 conftest fixture」的语义
        registry = FixtureRegistry(parent=base_registry)
        registry.register_all(mod)

        print(f"\n{path.name}")
        for name, fn in _cases(mod):
            total += 1
            status, info = _run_case(fn, registry)
            if status == "pass":
                passed += 1
                if verbose:
                    print(f"  PASS  {name}")
            elif status == "skip":
                skipped += 1
                print(f"  SKIP  {name}")
            else:
                failed += 1
                failures.append((f"{path.name}::{name}", info))
                print(f"  FAIL  {name}")

    dt = time.time() - t0
    print("\n" + "=" * 72)
    for title, info in failures:
        print(f"\n--- {title} ---")
        print(info.rstrip())
    print("=" * 72)
    print(f"{passed} passed, {failed} failed, {skipped} skipped  ({dt:.1f}s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

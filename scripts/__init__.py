"""可执行脚本集合。

做成包只是为了 ``from scripts._common import ...`` 在**任何**工作目录下
都能解析 —— 脚本自己还会把项目根塞进 ``sys.path``，
所以 ``python scripts/01_generate.py`` 和 ``python -m pytest``
两种入口都成立。
"""

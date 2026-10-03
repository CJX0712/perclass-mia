"""错误码与异常层次（作者：晨星）。

统一采用 E1xx~E5xx 命名空间，便于调用方按码分支处理与单测断言。
"""
from __future__ import annotations


class MIAError(Exception):
    """所有 MIAForge 异常的基类。"""

    code = "E000"


class ConfigError(MIAError):
    code = "E100"


class DataError(MIAError):
    code = "E200"


class TargetError(MIAError):
    code = "E300"


class AttackError(MIAError):
    code = "E400"


class PipelineError(MIAError):
    code = "E500"

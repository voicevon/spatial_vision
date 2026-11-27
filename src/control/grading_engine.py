#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单根绿芦笋分级规则引擎 (GradingEngine)
======================================
纯业务规则映射：将视觉测量输出 (effective_length_mm, diameter_mm, has_white_root)
按配置规则顺序匹配为分拣槽位等级 (grade: 1..N)，未命中或测量无效时返回剔除等级 (默认 0)。
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class GradeRule:
    """单条分级规则 (按配置顺序优先匹配)"""
    grade: int
    name: str
    min_len_mm: float = 0.0
    max_len_mm: Optional[float] = None
    min_diam_mm: float = 0.0
    max_diam_mm: Optional[float] = None
    allow_white_root: bool = True

    def matches(self, length_mm: float, diameter_mm: float, has_white_root: bool) -> bool:
        if has_white_root and not self.allow_white_root:
            return False
        if length_mm < self.min_len_mm:
            return False
        if self.max_len_mm is not None and length_mm > self.max_len_mm:
            return False
        if diameter_mm < self.min_diam_mm:
            return False
        if self.max_diam_mm is not None and diameter_mm > self.max_diam_mm:
            return False
        return True


@dataclass(frozen=True)
class GradeDecision:
    """分级判定结果"""
    grade: int
    name: str
    matched: bool
    reason: str = "OK"


class GradingEngine:
    """纯函数分级引擎 (零 IO、零硬件依赖)"""

    def __init__(self, rules: Iterable[GradeRule], default_reject_grade: int = 0):
        self.rules: List[GradeRule] = list(rules)
        self.default_reject_grade: int = int(default_reject_grade)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GradingEngine":
        default_reject = int(cfg.get("default_reject_grade", 0))
        rules: List[GradeRule] = []
        for item in cfg.get("rules") or []:
            rules.append(
                GradeRule(
                    grade=int(item["grade"]),
                    name=str(item.get("name", f"Grade-{item['grade']}")),
                    min_len_mm=float(item.get("min_len_mm", 0.0)),
                    max_len_mm=None if item.get("max_len_mm") is None else float(item["max_len_mm"]),
                    min_diam_mm=float(item.get("min_diam_mm", 0.0)),
                    max_diam_mm=None if item.get("max_diam_mm") is None else float(item["max_diam_mm"]),
                    allow_white_root=bool(item.get("allow_white_root", True)),
                )
            )
        return cls(rules=rules, default_reject_grade=default_reject)

    def evaluate(
        self,
        *,
        valid: bool,
        effective_length_mm: float,
        diameter_mm: float,
        has_white_root: bool = False,
    ) -> GradeDecision:
        if not valid:
            return GradeDecision(
                grade=self.default_reject_grade,
                name="剔除",
                matched=False,
                reason="INVALID_MEASUREMENT",
            )
        for rule in self.rules:
            if rule.matches(effective_length_mm, diameter_mm, has_white_root):
                return GradeDecision(
                    grade=rule.grade,
                    name=rule.name,
                    matched=True,
                    reason="OK",
                )
        return GradeDecision(
            grade=self.default_reject_grade,
            name="剔除",
            matched=False,
            reason="OUT_OF_SPEC",
        )

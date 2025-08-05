# schemas.py  (no pydantic)
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

_ALLOWED_TYPES = {"integer","int","numeric","float","string","object","categorical","categoric","date","datetime"}
_ALLOWED_OPS = {"==","!=",">=",">","<=","<"}

@dataclass
class ColumnRule:
    name: str
    type: str
    min: Optional[float]=None
    max: Optional[float]=None
    allowed_values: Optional[List[str]]=None
    outlier_threshold: Optional[float]=3.0

@dataclass
class ValidationRules:
    version: str
    date: str
    columns: List[ColumnRule] = field(default_factory=list)

@dataclass
class Condition:
    column: str
    operator: str
    value: Any

@dataclass
class Criterion:
    conditions: List[Condition] = field(default_factory=list)

@dataclass
class IdentificationRules:
    version: str
    date: str
    criteria: List[Criterion] = field(default_factory=list)

@dataclass
class EstimationRules:
    version: str
    date: str
    intercept: float = 0.0
    coefficients: Dict[str, float] = field(default_factory=dict)

def _ensure_keys(d: dict, keys: List[str], ctx: str):
    for k in keys:
        if k not in d:
            raise ValueError(f"Missing key '{k}' in {ctx}.")

def build_validation_rules(d: dict) -> ValidationRules:
    _ensure_keys(d, ["version","date","columns"], "validation rules")
    cols = []
    if not isinstance(d["columns"], list) or len(d["columns"]) == 0:
        raise ValueError("validation rules: 'columns' must be a non-empty list.")
    for i, c in enumerate(d["columns"]):
        _ensure_keys(c, ["name","type"], f"validation rules -> columns[{i}]")
        t = str(c["type"]).lower()
        if t not in _ALLOWED_TYPES:
            raise ValueError(f"validation rules: unsupported type '{t}' for column '{c['name']}'.")
        allowed = c.get("allowed_values")
        if allowed is not None:
            if not isinstance(allowed, list) or not all(isinstance(x, (str,int,float)) for x in allowed):
                raise ValueError(f"validation rules: 'allowed_values' must be a list of scalars for column '{c['name']}'.")
            allowed = [str(x) for x in allowed]
        cols.append(ColumnRule(
            name=str(c["name"]),
            type=t,
            min=float(c["min"]) if "min" in c and c["min"] is not None else None,
            max=float(c["max"]) if "max" in c and c["max"] is not None else None,
            allowed_values=allowed,
            outlier_threshold=float(c.get("outlier_threshold", 3.0)) if t in {"numeric","integer","float","int"} else None
        ))
    return ValidationRules(version=str(d["version"]), date=str(d["date"]), columns=cols)

def build_identification_rules(d: dict) -> IdentificationRules:
    _ensure_keys(d, ["version","date","criteria"], "identification rules")
    crits = []
    if not isinstance(d["criteria"], list):
        raise ValueError("identification rules: 'criteria' must be a list.")
    for i, crit in enumerate(d["criteria"]):
        _ensure_keys(crit, ["conditions"], f"identification rules -> criteria[{i}]")
        conds = []
        if not isinstance(crit["conditions"], list) or len(crit["conditions"]) == 0:
            raise ValueError(f"identification rules: 'conditions' must be a non-empty list in criteria[{i}].")
        for j, cond in enumerate(crit["conditions"]):
            _ensure_keys(cond, ["column","operator","value"], f"identification rules -> criteria[{i}].conditions[{j}]")
            op = str(cond["operator"])
            if op not in _ALLOWED_OPS:
                raise ValueError(f"identification rules: unsupported operator '{op}'.")
            conds.append(Condition(column=str(cond["column"]), operator=op, value=cond["value"]))
        crits.append(Criterion(conditions=conds))
    return IdentificationRules(version=str(d["version"]), date=str(d["date"]), criteria=crits)

def build_estimation_rules(d: dict) -> EstimationRules:
    _ensure_keys(d, ["version","date"], "estimation rules")
    coefs = d.get("coefficients", {})
    if not isinstance(coefs, dict):
        raise ValueError("estimation rules: 'coefficients' must be a mapping.")
    coeffs_norm = {}
    for k, v in coefs.items():
        try:
            coeffs_norm[str(k)] = float(v)
        except Exception:
            raise ValueError(f"estimation rules: coefficient for '{k}' is not numeric.")
    intercept = float(d.get("intercept", 0.0))
    return EstimationRules(version=str(d["version"]), date=str(d["date"]), intercept=intercept, coefficients=coeffs_norm)

def validate_validation_rules_dict(d: dict) -> dict:
    _ = build_validation_rules(d)
    return d

def validate_identification_rules_dict(d: dict) -> dict:
    _ = build_identification_rules(d)
    return d

def validate_estimation_rules_dict(d: dict) -> dict:
    _ = build_estimation_rules(d)
    return d

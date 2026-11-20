"""Cross-contract drift guards.

Every artifact in `benchmarks/` states part of one contract in more than one
place: a JSON schema and a Python module, a docstring claim and a test, a
constant in Class A and its use in Class B. Each duplication is a place the
contract can fork silently.

Two forks have already been caught and closed by guards like these — the
identity schema against `IDENTITY_COMPONENTS`, and the documented failure modes
against their pinning tests. This module holds the remainder.

Every test here passes today. They exist as regression guards, not fixes.

Run explicitly (``benchmarks`` is not in ``pytest.ini`` testpaths):

    pytest benchmarks/ -q
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks import gold_metrics, process_metrics  # noqa: E402

ACTION_LOG_SCHEMA = json.loads((REPO / "benchmarks/AGENT_ACTION_LOG.schema.json").read_text())
ACTION_PROPS = ACTION_LOG_SCHEMA["$defs"]["action"]["properties"]
ACTION_CLASSES = frozenset(ACTION_PROPS["action_class"]["enum"])
RESULT_STATUSES = frozenset(ACTION_PROPS["result_status"]["enum"])


# --- Python constants vs schema enums ---------------------------------------


def test_every_action_class_constant_exists_in_the_schema_enum():
    """A typo like `mutated` would make a metric silently match nothing."""
    for name in ("MUTATING_CLASSES", "READING_CLASSES", "VERIFYING_CLASSES"):
        values = getattr(process_metrics, name)
        assert values, f"{name} is empty"
        assert values <= ACTION_CLASSES, f"{name} has values outside the schema enum: {values - ACTION_CLASSES}"


def test_invalid_statuses_exist_in_the_schema_enum():
    assert process_metrics.INVALID_STATUSES <= RESULT_STATUSES


def test_action_classes_no_metric_consumes_are_deliberate_not_forgotten():
    """`execute`, `escalate`, `other` are intentionally unconsumed by Class A.

    `escalate` is reserved for Class D. `execute` and `other` are catch-alls that
    no Class A metric should special-case — pinning that here means adding a new
    enum value forces a decision instead of silently doing nothing.
    """
    consumed = (
        process_metrics.MUTATING_CLASSES
        | process_metrics.READING_CLASSES
        | process_metrics.VERIFYING_CLASSES
    )
    assert ACTION_CLASSES - consumed == {"execute", "escalate", "other"}


# --- Class A / Class B agreement --------------------------------------------


def test_class_a_and_class_b_share_one_definition_of_a_mutation():
    """Not merely equal — the same object.

    Equality would still permit a fork later. Identity means gold_metrics imports
    the constant rather than restating it, so the two classes cannot come to
    measure different populations of the same log.
    """
    assert gold_metrics.MUTATING_CLASSES is process_metrics.MUTATING_CLASSES


def test_no_metric_module_declares_a_dead_taxonomy_constant():
    """An unused constant that looks authoritative invites use as if it were wired."""
    for module in (process_metrics, gold_metrics):
        source = Path(module.__file__).read_text()
        for name in dir(module):
            if not name.isupper() or name.startswith("_"):
                continue
            value = getattr(module, name)
            if not isinstance(value, (frozenset, set)):
                continue
            uses = source.count(name)
            assert uses > 1, f"{module.__name__}.{name} is declared but never used"


# --- schema fields vs consumers ---------------------------------------------


def test_every_class_b_schema_field_has_a_consumer():
    """`tests_run` and `lines_changed` were added for Class B; verify they landed."""
    gold_source = (REPO / "benchmarks/gold_metrics.py").read_text()
    assert "tests_run" in ACTION_PROPS and "tests_run" in gold_source
    assert "lines_changed" in ACTION_PROPS and "lines_changed" in gold_source


def _code_identifiers(path: Path) -> set[str]:
    """Every name appearing in executable code, with docstrings and comments excluded.

    Substring-matching the file would trip on any mention in prose — a metric
    docstring that *points at* a field reads identically to one that consumes it.
    Walking the AST distinguishes the two.
    """
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)  # dict keys and .get() arguments are string constants
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    docstrings = {
        ast.get_docstring(n)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return names - {d for d in docstrings if d}


def test_class_d_fields_remain_reserved_with_no_consumer():
    """The inverse guard: `confidence` is present for retrofit reasons only.

    If a metric starts reading it without the calibration protocol being decided,
    this fails and forces that decision to be explicit. Checked against code, not
    prose — both metric modules legitimately *mention* confidence in docstrings.
    """
    assert "confidence" in ACTION_PROPS
    assert "CLASS D" in ACTION_PROPS["confidence"]["description"]
    for path in ("benchmarks/process_metrics.py", "benchmarks/gold_metrics.py"):
        assert "confidence" not in _code_identifiers(REPO / path)


# --- no module promotes a gate ----------------------------------------------


def test_no_metric_module_defines_a_numeric_constant():
    """Diagnostics, not gates. A module-level number here would be a promoted threshold.

    `confidence.py` is exempt by design: it implements the bound a caller compares
    against, and takes epsilon as a required parameter rather than choosing one.
    """
    for path in ("benchmarks/process_metrics.py", "benchmarks/gold_metrics.py"):
        tree = ast.parse((REPO / path).read_text())
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                target = ast.unparse(node.targets[0] if isinstance(node, ast.Assign) else node.target)
                raise AssertionError(f"{path} defines numeric constant {target} = {value.value}")


def test_confidence_module_requires_an_explicit_epsilon():
    """No default epsilon anywhere — choosing one is the caller's decision to justify."""
    tree = ast.parse((REPO / "benchmarks/confidence.py").read_text())
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for arg, default in zip(
            node.args.kwonlyargs, node.args.kw_defaults or [], strict=False
        ):
            if arg.arg == "epsilon":
                found = True
                assert default is None, "epsilon must not have a default"
        for arg in node.args.args:
            if arg.arg == "epsilon":
                found = True
    assert found, "no epsilon parameter found — did the API change?"

"""Canonical table representation — nano.table.v0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nanoscribe.artifacts.errors import artifact_fail
from nanoscribe.artifacts.validation import (
    require_mapping,
    require_nonempty_string,
    require_schema_version,
    require_string_list,
)

TABLE_SCHEMA_VERSION = "nano.table.v0"
ALLOWED_COLUMN_TYPES = frozenset({"string", "number", "boolean"})


@dataclass(frozen=True, slots=True)
class TableColumn:
    key: str
    label: str
    data_type: str = "string"

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "data_type": self.data_type}

    @classmethod
    def from_dict(cls, data: object, *, path: str) -> TableColumn:
        mapping = require_mapping(data, path)
        data_type = mapping.get("data_type", "string")
        if not isinstance(data_type, str) or data_type not in ALLOWED_COLUMN_TYPES:
            allowed = ", ".join(sorted(ALLOWED_COLUMN_TYPES))
            artifact_fail("invalid_enum", f"data_type must be one of: {allowed}", f"{path}.data_type")
        return cls(
            key=require_nonempty_string(mapping.get("key", ""), f"{path}.key"),
            label=require_nonempty_string(mapping.get("label", ""), f"{path}.label"),
            data_type=data_type,
        )


@dataclass(frozen=True, slots=True)
class TableSpec:
    title: str | None
    columns: tuple[TableColumn, ...]
    rows: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TABLE_SCHEMA_VERSION,
            "title": self.title,
            "columns": [column.to_dict() for column in self.columns],
            "rows": [list(row) for row in self.rows],
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> TableSpec:
        mapping = require_mapping(data, path)
        require_schema_version(mapping.get("schema_version"), TABLE_SCHEMA_VERSION, f"{path}.schema_version")
        title = mapping.get("title")
        if title is not None and not isinstance(title, str):
            artifact_fail("type_error", "title must be a string or null", f"{path}.title")
        columns_raw = mapping.get("columns")
        if not isinstance(columns_raw, list) or not columns_raw:
            artifact_fail("type_error", "columns must be a non-empty array", f"{path}.columns")
        columns = tuple(
            TableColumn.from_dict(column, path=f"{path}.columns[{index}]")
            for index, column in enumerate(columns_raw)
        )
        keys = {column.key for column in columns}
        if len(keys) != len(columns):
            artifact_fail("duplicate_key", "column keys must be unique", f"{path}.columns")
        rows_raw = mapping.get("rows", [])
        if not isinstance(rows_raw, list):
            artifact_fail("type_error", "rows must be an array", f"{path}.rows")
        width = len(columns)
        rows: list[tuple[str, ...]] = []
        for index, row in enumerate(rows_raw):
            row_path = f"{path}.rows[{index}]"
            if not isinstance(row, list):
                artifact_fail("type_error", "each row must be an array", row_path)
            if len(row) != width:
                artifact_fail(
                    "invalid_row_width",
                    f"row width {len(row)} != column count {width}",
                    row_path,
                )
            if any(not isinstance(cell, str) for cell in row):
                artifact_fail("type_error", "row cells must be strings", row_path)
            rows.append(tuple(row))
        return cls(title=title, columns=columns, rows=tuple(rows))

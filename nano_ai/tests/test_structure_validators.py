"""Tests for the P4 structure validators (Markdown table, Mermaid diagram)."""

from __future__ import annotations

from nano_ai.structure_validators import (
    StructureValidation,
    validate_markdown_table,
    validate_mermaid,
)


class TestStructureValidationReward:
    def test_valid_reward_is_one(self) -> None:
        assert StructureValidation(True, "ok").reward == 1.0

    def test_invalid_reward_is_zero(self) -> None:
        assert StructureValidation(False, "nope").reward == 0.0

    def test_is_frozen(self) -> None:
        result = StructureValidation(True, "ok")
        try:
            result.valid = False  # type: ignore[misc]
        except Exception:
            pass
        else:
            raise AssertionError("StructureValidation should be immutable")


class TestValidateMarkdownTable:
    def test_well_formed_table_is_valid(self) -> None:
        text = (
            "| Name | Value |\n"
            "| --- | --- |\n"
            "| medication | ibuprofen |\n"
            "| allergy | none |\n"
        )
        result = validate_markdown_table(text)
        assert result.valid is True
        assert result.reason == "ok"

    def test_table_without_leading_trailing_pipes_is_valid(self) -> None:
        text = "Name | Value\n--- | ---\nmedication | ibuprofen\n"
        assert validate_markdown_table(text).valid is True

    def test_alignment_colons_are_accepted(self) -> None:
        text = "| A | B | C |\n| :--- | :---: | ---: |\n| 1 | 2 | 3 |\n"
        assert validate_markdown_table(text).valid is True

    def test_single_line_is_invalid(self) -> None:
        result = validate_markdown_table("| Name | Value |")
        assert result.valid is False
        assert "fewer than 2" in result.reason

    def test_empty_string_is_invalid(self) -> None:
        assert validate_markdown_table("").valid is False

    def test_missing_pipe_in_header_is_invalid(self) -> None:
        text = "Name Value\n--- ---\n"
        result = validate_markdown_table(text)
        assert result.valid is False
        assert "'|'" in result.reason

    def test_missing_separator_row_is_invalid(self) -> None:
        text = "| Name | Value |\n| medication | ibuprofen |\n"
        result = validate_markdown_table(text)
        assert result.valid is False
        assert "separator" in result.reason

    def test_separator_row_wrong_column_count_is_invalid(self) -> None:
        text = "| Name | Value |\n| --- |\n| medication | ibuprofen |\n"
        result = validate_markdown_table(text)
        assert result.valid is False
        assert "separator column count" in result.reason

    def test_data_row_wrong_column_count_is_invalid(self) -> None:
        text = "| Name | Value |\n| --- | --- |\n| medication |\n"
        result = validate_markdown_table(text)
        assert result.valid is False
        assert "row 3" in result.reason

    def test_separator_row_without_dashes_is_invalid(self) -> None:
        text = "| Name | Value |\n| xxx | yyy |\n| medication | ibuprofen |\n"
        result = validate_markdown_table(text)
        assert result.valid is False
        assert "not a valid separator row" in result.reason

    def test_header_only_no_data_rows_is_valid(self) -> None:
        text = "| Name | Value |\n| --- | --- |\n"
        assert validate_markdown_table(text).valid is True

    def test_blank_lines_are_ignored(self) -> None:
        text = "| Name | Value |\n\n| --- | --- |\n\n| medication | ibuprofen |\n"
        assert validate_markdown_table(text).valid is True


class TestValidateMermaid:
    def test_flowchart_is_valid(self) -> None:
        text = "flowchart TD\n  A[Start] --> B{Decision}\n  B --> C[End]\n"
        result = validate_mermaid(text)
        assert result.valid is True
        assert result.reason == "ok"

    def test_graph_is_valid(self) -> None:
        text = "graph LR\n  A --> B\n  B --> C\n"
        assert validate_mermaid(text).valid is True

    def test_sequence_diagram_is_valid(self) -> None:
        text = "sequenceDiagram\n  Alice->>Bob: Hello Bob\n  Bob-->>Alice: Hi Alice\n"
        assert validate_mermaid(text).valid is True

    def test_class_diagram_is_valid(self) -> None:
        text = "classDiagram\n  Animal <|-- Duck\n  Animal : +String name\n"
        assert validate_mermaid(text).valid is True

    def test_state_diagram_v2_is_valid(self) -> None:
        text = "stateDiagram-v2\n  [*] --> Idle\n  Idle --> Running\n"
        assert validate_mermaid(text).valid is True

    def test_empty_string_is_invalid(self) -> None:
        result = validate_mermaid("")
        assert result.valid is False
        assert "empty" in result.reason

    def test_comment_only_is_invalid(self) -> None:
        result = validate_mermaid("%% just a comment\n")
        assert result.valid is False
        assert "empty" in result.reason

    def test_unknown_diagram_type_is_invalid(self) -> None:
        text = "notADiagram\n  A --> B\n"
        result = validate_mermaid(text)
        assert result.valid is False
        assert "not a known diagram type" in result.reason

    def test_declaration_with_no_body_is_invalid(self) -> None:
        result = validate_mermaid("flowchart TD\n")
        assert result.valid is False
        assert "no body" in result.reason

    def test_unbalanced_brackets_is_invalid(self) -> None:
        text = "flowchart TD\n  A[Start --> B{Decision}\n"
        result = validate_mermaid(text)
        assert result.valid is False
        assert "unbalanced" in result.reason

    def test_unbalanced_parens_is_invalid(self) -> None:
        text = "flowchart TD\n  A(Start --> B(End\n"
        result = validate_mermaid(text)
        assert result.valid is False
        assert "unbalanced" in result.reason

    def test_bare_diagram_keyword_without_direction_is_valid(self) -> None:
        text = "flowchart\n  A --> B\n"
        assert validate_mermaid(text).valid is True

    def test_leading_comment_before_declaration_is_ignored(self) -> None:
        text = "%% comment\nflowchart TD\n  A --> B\n"
        assert validate_mermaid(text).valid is True

    def test_pie_chart_is_valid(self) -> None:
        text = 'pie title Pets\n  "Dogs" : 40\n  "Cats" : 60\n'
        assert validate_mermaid(text).valid is True

    def test_diagram_type_prefix_that_is_not_a_real_keyword_is_invalid(self) -> None:
        text = "graphical TD\n  A --> B\n"
        result = validate_mermaid(text)
        assert result.valid is False
        assert "not a known diagram type" in result.reason

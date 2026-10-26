"""Tests for dynacell.preprocess.workflow."""

from dynacell.preprocess.workflow import extract_numeric_part, is_target_workflow


class TestExtractNumericPart:
    """Tests for extract_numeric_part."""

    def test_pipeline_prefix(self):
        """Strips 'Pipeline ' prefix and brackets."""
        assert extract_numeric_part("[Pipeline 4.1]") == "4.1"

    def test_bare_number(self):
        """Passes through a bare numeric string."""
        assert extract_numeric_part("4.2") == "4.2"

    def test_quoted_brackets(self):
        """Strips surrounding quotes and brackets."""
        assert extract_numeric_part("['Pipeline 4']") == "4"

    def test_integer_id(self):
        """Handles integer-only workflow ID."""
        assert extract_numeric_part("4") == "4"


class TestIsTargetWorkflow:
    """Tests for is_target_workflow."""

    def test_positive_match(self):
        """Returns True when workflow ID is in targets."""
        assert is_target_workflow("[Pipeline 4.1]", ["4", "4.1", "4.2"])

    def test_negative_match(self):
        """Returns False when workflow ID is not in targets."""
        assert not is_target_workflow("[Pipeline 3]", ["4", "4.1"])

"""Tests for PHATE visualization functions."""

import pandas as pd
import plotly.graph_objects as go

from visualizer.visualization import create_phate_figure


class TestCreatePhateFigure:
    """Tests for create_phate_figure function."""

    def test_create_figure_basic(self, sample_tracks_df):
        """Test basic figure creation."""
        fig = create_phate_figure(sample_tracks_df, color_by="annotation")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.xaxis.title.text == "PHATE1"
        assert fig.layout.yaxis.title.text == "PHATE2"

    def test_color_by_annotation(self, sample_tracks_df):
        """Test coloring by annotation."""
        fig = create_phate_figure(sample_tracks_df, color_by="annotation")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        trace_names = [trace.name for trace in fig.data]
        assert any("uninfected" in name.lower() for name in trace_names)
        assert any("infected" in name.lower() for name in trace_names)

    def test_color_by_time(self, sample_tracks_df):
        """Test coloring by time."""
        fig = create_phate_figure(sample_tracks_df, color_by="time")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        has_colorscale = any(
            hasattr(trace, "marker") and hasattr(trace.marker, "colorscale")
            for trace in fig.data
        )
        assert has_colorscale

    def test_color_by_track_id(self, sample_tracks_df):
        """Test coloring by track ID."""
        sample_tracks_df["track_key"] = sample_tracks_df["track_id"].astype(str)
        selected_tracks = ["1", "2"]

        fig = create_phate_figure(
            sample_tracks_df, color_by="track_id", selected_tracks=selected_tracks
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        trace_names = [trace.name for trace in fig.data]
        assert "1" in trace_names or "2" in trace_names

    def test_color_by_dataset(self, sample_multi_dataset_tracks_df):
        """Test coloring by dataset."""
        sample_multi_dataset_tracks_df["track_key"] = sample_multi_dataset_tracks_df[
            "track_id"
        ].astype(str)

        fig = create_phate_figure(sample_multi_dataset_tracks_df, color_by="dataset")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        trace_names = [trace.name for trace in fig.data]
        assert any(str(ds) in str(name) for ds in [0, 1] for name in trace_names)

    def test_selected_tracks_highlighting(self, sample_tracks_df):
        """Test that selected tracks are highlighted."""
        sample_tracks_df["track_key"] = sample_tracks_df["track_id"].astype(str)
        selected_tracks = ["1"]

        fig = create_phate_figure(
            sample_tracks_df, color_by="annotation", selected_tracks=selected_tracks
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        has_background = any("background" in str(trace.name) for trace in fig.data)
        has_selected = any("selected" in str(trace.name) for trace in fig.data)

        assert has_background or has_selected

    def test_show_trajectories(self, sample_tracks_df):
        """Test trajectory lines are added."""
        sample_tracks_df["track_key"] = sample_tracks_df["track_id"].astype(str)
        selected_tracks = ["1", "2"]

        fig = create_phate_figure(
            sample_tracks_df,
            color_by="annotation",
            selected_tracks=selected_tracks,
            show_trajectories=True,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        has_lines = any(
            hasattr(trace, "mode") and "lines" in trace.mode for trace in fig.data
        )
        assert has_lines

    def test_highlight_timepoint(self, sample_tracks_df):
        """Test timepoint highlighting."""
        sample_tracks_df["track_key"] = sample_tracks_df["track_id"].astype(str)
        selected_tracks = ["1", "2"]

        fig = create_phate_figure(
            sample_tracks_df,
            color_by="annotation",
            selected_tracks=selected_tracks,
            highlight_timepoint=1,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        trace_names = [trace.name for trace in fig.data]
        assert any("t=" in str(name) for name in trace_names)

    def test_selected_values_filtering(self, sample_tracks_df):
        """Test filtering by selected annotation values."""
        fig = create_phate_figure(
            sample_tracks_df, color_by="annotation", selected_values=["infected"]
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        trace_names = [trace.name for trace in fig.data]
        trace_names_lower = [str(name).lower() for name in trace_names]

        assert any("infected" in name for name in trace_names_lower)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(
            columns=["PHATE1", "PHATE2", "annotation", "t", "track_id"]
        )

        fig = create_phate_figure(empty_df, color_by="annotation")

        assert isinstance(fig, go.Figure)

    def test_figure_layout_properties(self, sample_tracks_df):
        """Test that figure has expected layout properties."""
        fig = create_phate_figure(sample_tracks_df, color_by="annotation")

        assert fig.layout.title.text is not None
        assert "PHATE" in fig.layout.title.text
        assert fig.layout.height == 600
        assert fig.layout.template == "plotly_white"
        assert fig.layout.hovermode == "closest"

    def test_multiple_timepoints(self, sample_tracks_df):
        """Test with data containing multiple timepoints."""
        assert sample_tracks_df["t"].nunique() > 1

        fig = create_phate_figure(sample_tracks_df, color_by="time")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_customdata_includes_required_fields(self, sample_tracks_df):
        """Test that customdata includes required fields for hover."""
        sample_tracks_df["track_key"] = sample_tracks_df["track_id"].astype(str)

        fig = create_phate_figure(sample_tracks_df, color_by="annotation")

        for trace in fig.data:
            if hasattr(trace, "customdata") and trace.customdata is not None:
                assert len(trace.customdata) > 0
                assert trace.customdata.shape[1] >= 4

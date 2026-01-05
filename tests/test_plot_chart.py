import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import plot_chart


def test_line_spec_validation_requires_axes():
    plot_chart.st.session_state["chart_spec"] = []

    outcome = plot_chart.make_chart_spec(
        chart_type="line",
        data=[{"time_bucket": "2025-10-01", "value": 10}],
        title="Example",
        x=None,
        y="value",
    )

    assert outcome["error"] == "MISSING_AXIS"
    assert plot_chart.st.session_state["chart_spec"] == []


def test_bar_spec_structure():
    plot_chart.st.session_state["chart_spec"] = []
    plot_chart.drain_chart_buffer()

    outcome = plot_chart.make_chart_spec(
        chart_type="bar",
        data=[{"category": "NCR", "tickets": 42}],
        title="Tickets by Region",
        x="category",
        y="tickets",
    )

    assert outcome["status"] == "chart_spec_added"
    assert outcome["spec"]["type"] == "bar"
    assert outcome["spec"]["data"][0]["category"] == "NCR"
    stored = plot_chart.st.session_state["chart_spec"]
    assert stored
    assert stored[0]["spec"]["type"] == "bar"
    assert stored[0]["spec"]["data"][0]["category"] == "NCR"


def test_buffer_captures_specs_outside_streamlit_context():
    plot_chart.st.session_state["chart_spec"] = []
    plot_chart.drain_chart_buffer()

    plot_chart.make_chart_spec(
        chart_type="line",
        data=[{"date": "2025-10-01", "value": 3}],
        title="Example",
        x="date",
        y="value",
    )

    buffered = plot_chart.drain_chart_buffer()
    assert buffered
    assert buffered[0]["spec"]["type"] == "line"


def test_normalize_chart_state_accepts_raw_spec():
    raw = {
        "type": "line",
        "data": [{"time_bucket": "2025-10-01", "value": 10}],
        "xField": "time_bucket",
        "yField": "value",
    }

    normalized = plot_chart.normalize_chart_state(raw)
    assert normalized is not None
    assert normalized["spec"]["type"] == "line"

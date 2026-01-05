from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import uuid

try:  # pragma: no cover - streamlit optional for tests
    import streamlit as st
except ImportError:  # pragma: no cover
    class _SessionState(dict):
        pass

    class _StreamlitStub:
        def __init__(self) -> None:
            self.session_state: Dict[str, Any] = {}

        def plotly_chart(self, *args: Any, **kwargs: Any) -> None:
            return None

    st = _StreamlitStub()


_CHART_SPEC_BUFFER: List[Dict[str, Any]] = []
_CHART_BUFFER_LOCK = Lock()


def _buffer_chart_spec(entry: Dict[str, Any]) -> None:
    with _CHART_BUFFER_LOCK:
        _CHART_SPEC_BUFFER.append(entry)


def drain_chart_buffer() -> List[Dict[str, Any]]:
    with _CHART_BUFFER_LOCK:
        buffered = list(_CHART_SPEC_BUFFER)
        _CHART_SPEC_BUFFER.clear()
    return buffered


def _chart_collection() -> List[Dict[str, Any]]:
    charts = st.session_state.get("chart_spec")
    if not isinstance(charts, list):
        charts = []
        st.session_state["chart_spec"] = charts
    return charts


def _record_chart_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist the chart spec for UI rendering and return it so upstream
    callers (LM/tooling) can relay the spec back through their outputs.
    """
    entry = {"spec": spec}
    _buffer_chart_spec(entry)
    _chart_collection().append(entry)
    return {"status": "chart_spec_added", "spec": spec}


def make_chart_spec(
    chart_type: str,
    data: List[Dict[str, Any]],
    *,
    title: str = "",
    x: Optional[str] = None,
    y: Optional[str] = None,
    series: Optional[str] = None,
    label: Optional[str] = None,
    value: Optional[str] = None,
    orientation: Optional[str] = None,
    key: str, 
) -> Dict[str, Any]:
    """
    Build a serialisable chart specification for Streamlit rendering.

    Args:
        chart_type: One of ``"line"``, ``"bar"``, or ``"pie"``.
        data: List of dictionaries where each entry represents a row for the chart.
        title: Optional chart title.
        x: Column to plot on the X axis (required for line and bar charts).
        y: Column to plot on the Y axis (required for line and bar charts).
        series: Optional grouping column for line charts.
        label: Column used for pie chart labels.
        value: Column used for pie chart values.
        orientation: For bar charts, either ``"vertical"`` (default) or ``"horizontal"``.
        key: Unique key to identify the chart.

    Returns:
        Dictionary with ``spec`` key describing the chart. On validation failure, returns a dictionary
        with ``error``, ``message``, and optional ``hint`` to help the agent recover.
    """
    supported_types = {"line", "bar", "pie"}
    if not chart_type or chart_type not in supported_types:
        return {
            "error": "INVALID_CHART_TYPE",
            "message": f"Unsupported chart_type '{chart_type}'.",
            "hint": "Use one of: line, bar, pie.",
        }
    
    if not key or not isinstance(key, str):
        return {
            "error": "MISSING_KEY",
            "message": "A unique string key must be provided to identify the chart.",
            "hint": "Set the 'key' parameter to a non-empty string value.",
        }

    if not isinstance(data, list):
        return {
            "error": "INVALID_DATA",
            "message": "Chart data must be provided as a list of dictionaries.",
            "hint": "Build data like [{'category': 'A', 'value': 10}, ...].",
        }
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            return {
                "error": "INVALID_ROW",
                "message": f"Chart data row at index {idx} must be a dictionary.",
                "hint": "Ensure every data point is a dict mapping field names to values.",
            }

    if chart_type == "line":
        if not x or not y:
            return {
                "error": "MISSING_AXIS",
                "message": "Line chart requires both 'x' and 'y' fields.",
                "hint": "Provide x and y keys when building the chart spec.",
            }
        spec = {
            "type": "line",
            "title": title,
            "data": data,
            "xField": x,
            "yField": y,
            "seriesField": series,
            "key": key,
        }
        return _record_chart_spec(spec)

    if chart_type == "bar":
        if not x or not y:
            return {
                "error": "MISSING_AXIS",
                "message": "Bar chart requires both 'x' and 'y' fields.",
                "hint": "Provide x and y keys when building the chart spec.",
            }
        if orientation and orientation not in {"vertical", "horizontal"}:
            return {
                "error": "INVALID_ORIENTATION",
                "message": f"Unsupported orientation '{orientation}'.",
                "hint": "Choose 'vertical' or 'horizontal' when specifying orientation.",
            }
        spec = {
            "type": "bar",
            "title": title,
            "data": data,
            "xField": x,
            "yField": y,
            "orientation": orientation or "vertical",
            "key": key,
        }
        return _record_chart_spec(spec)

    if chart_type == "pie":
        if not label or not value:
            return {
                "error": "MISSING_FIELDS",
                "message": "Pie chart requires both 'label' and 'value' fields.",
                "hint": "Set label and value keys to identify categories and their measures.",
            }
        spec = {
            "type": "pie",
            "title": title,
            "data": data,
            "labelField": label,
            "valueField": value,
            "key": key,
        }
        return _record_chart_spec(spec)

    raise ValueError(f"Unsupported chart_type '{chart_type}'.")


def normalize_chart_state(chart_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not chart_state:
        return None

    if "spec" in chart_state and isinstance(chart_state["spec"], dict):
        return {"spec": dict(chart_state["spec"])}

    if "type" in chart_state and "data" in chart_state:
        return {"spec": dict(chart_state)}

    return None


def _to_dataframe(data: List[Dict[str, Any]]):
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def render_chart(chart_state: Dict[str, Any]) -> None: # or however you build your figure
    normalized = normalize_chart_state(chart_state)
    if not normalized:
        return

    spec = normalized["spec"]
    if not spec or "type" not in spec:
        return

    data_rows = spec.get("data", [])
    if not data_rows:
        return

    df = _to_dataframe(data_rows)
    if df.empty:
        return

    chart_type = spec.get("type")
    title = spec.get("title", "")
    
    base_key = spec.get("key")
    if "rendered_keys" not in st.session_state:
        st.session_state.rendered_keys = set()

    unique_key = base_key
    if base_key in st.session_state.rendered_keys:
        unique_key = f"{base_key}_{uuid.uuid4().hex[:6]}"

    st.session_state.rendered_keys.add(unique_key)

    if chart_type == "line":
        x_field = spec.get("xField")
        y_field = spec.get("yField")
        series_field = spec.get("seriesField")
        if x_field not in df.columns or y_field not in df.columns:
            return
        if series_field and series_field in df.columns:
            fig = px.line(df, x=x_field, y=y_field, color=series_field, title=title)
        else:
            fig = px.line(df, x=x_field, y=y_field, title=title)
        st.plotly_chart(fig, use_container_width=True, key = unique_key)
        return

    if chart_type == "bar":
        x_field = spec.get("xField")
        y_field = spec.get("yField")
        orientation = spec.get("orientation", "vertical")
        if x_field not in df.columns or y_field not in df.columns:
            return
        if orientation == "horizontal":
            fig = px.bar(df, x=y_field, y=x_field, orientation="h", title=title)
        else:
            fig = px.bar(df, x=x_field, y=y_field, title=title)
        st.plotly_chart(fig, use_container_width=True, key= unique_key)
        return

    if chart_type == "pie":
        label_field = spec.get("labelField")
        value_field = spec.get("valueField")
        if label_field not in df.columns or value_field not in df.columns:
            return
        fig = px.pie(df, names=label_field, values=value_field, title=title)
        st.plotly_chart(fig, use_container_width=True, key= unique_key)
        return

    # Unsupported chart type - fail silently for UI safety.

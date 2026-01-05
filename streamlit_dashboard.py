import streamlit as st
import pandas as pd
import dspy
import io
from typing import List, Optional, Any, Dict, Tuple, Set
import os
import json
import logging
from pathlib import Path
from collections import defaultdict
from itertools import zip_longest
from contextlib import nullcontext
import zipfile
import datetime
import plotly.express as px
import numpy as np
import product_category
import requests
import tools.calculate_metrics as calculate_metrics
import tools.trend_breakdown as trend_breakdown
import tools.plot_chart as plot_chart
import tools.resolve_filter as resolve_filter
import tools.catalog_loader as catalog_loader
import tools.get_examples as get_examples
import tools.ticket_summariser as ticket_summariser
from dspy.streaming import StreamListener, StatusMessage, StreamResponse, StatusMessageProvider

try:
    from streamlit_theme import st_theme
except ImportError:
    st_theme = None

DATA_URL = st.secrets.get("DATA_URL")
FOLDER_ID = st.secrets.get("FOLDER_ID")

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
AGENT_LOG_FILE = LOG_DIR / "agent_steps.log"

agent_logger = logging.getLogger("agent_steps")
if not agent_logger.handlers:
    handler = logging.FileHandler(AGENT_LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    agent_logger.addHandler(handler)
    agent_logger.setLevel(logging.INFO)

import hashlib

def df_fingerprint(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

def init_agent_state(prefix: str):
    defaults = {
        "user_messages": [],
        "assistant_messages": [],
        "chart_spec": [],
        "agent_last_step_trace": "",
        "agent_tool_events": [],
        "filters_created": [],
        "full_query_text": "",
        "vector_scope_hash": None,
    }

    for k, v in defaults.items():
        key = f"{prefix}_{k}"
        if key not in st.session_state:
            st.session_state[key] = v

init_agent_state("global")
init_agent_state("filtered")



class MainReactAgent(dspy.Signature):
    """
    SYSTEM ROLE:
    You are Support Insights Analyst of Country Delight, an analytical assistant for analysis customer support tickets, refunds,
    and quality issues. You answer questions using ticket data, aggregations, and tool outputs. You also have memory of the previous messages
    in this conversation to provide context-aware responses. While doing back to back conversation, always refer to the previous messages, to find relevant context. 
    
    DATABASE DESCRIPTION:
    The ticket database contains customer support tickets with the following fields:
        - complaint_number: Unique identifier for each ticket
        - city: The city from where the ticket was raised
        - region: The region where the ticket was raised
        - created_date: The date the ticket was created
        - refund_count_in_15_days: The number of refunds issued to this ticket's customer in the last 15 days
        - product: The product related to the ticket
        - concern_type: The type of ticket type, Literal values: 'Complaint', 'Feedback', 'Request Action', 'Request Information'
        - level_1_classification: The first level classification of the ticket, Literal values: 'Delivery & Fulfillment', 'Product / Quality', 'Billing, Offers, Payments', 'User Account and Membership', 'Orders', 'Customer Support Issue', 'Returns and Replacement'
        - level_2_classification: The second level classification of the ticket eg. 'Product Not Fresh / Expired / Rotten / Infested', 'Partial / Missing Items at Delivery' etc.
        - expanded_description: A detailed description of the ticket
        - customer_issue: The specific issue raised by the customer
        - root_cause(optional): The underlying cause of the ticket
        - resolution_provided_summary: A summary of the resolution provided to the customer
        - product_category: The category of the product related to the ticket, Literal values: 'Fruits', 'Vegetables', 'Dairy', 'Staples', 'Coconut Water', 'Snacks'

    STRICT RULES:
    1. Always use tools to get numbers, trends, and examples. Never invent numeric values.
    2. Always use resolved filters (filter_id, canonical SKUs, date ranges) from tools.
       Do not guess SKU names, regions, or classification labels.
    3. Prefer the examples tool with text hints (text_query/theme_hint/text_column) when users ask "what are customers saying" or reference specific themes. Always pass the existing filter_id so semantic search runs within the structured slice.
    4. Use the summary returned by get_examples (if present) in your narrative; only list individual complaints (max 5) when the user explicitly requests examples.
    5. When describing trends, clearly mention the time window returned by tools.
    6. For comparisons (week-over-week, etc.), only use values returned by the metrics tool.
       Never estimate percentages yourself.
    7. If a chart is relevant, you MUST request a chart spec from the chart tool and include it
       as `chart` in the final answer.
    8. If the user request is underspecified, assume they want:
       (a) top issues, (b) trend over time, (c) short explanation of why.
    9. IMPORTANT: Always use resolve_filter tool before calling create_filter_record or update_filter_record.
    
    TOOL INSTRUCTIONS:
    - While using 'resolve_filters', always pass all of the requested arguments, if they are none, just pass an empty list, otherwise the tool would return nothing. 
    - Always use calculate_metrics if the user asks for numbers, trends or comparisons. Always use this after trend_breakdown. 
    

    RESPONSE STYLE:
    - Speak conversationally while staying factual and action-focused.
    - Call out charts you generated so the user knows visuals are available.
    - If data is limited or a tool fails, explain the gap and suggest next steps instead of guessing.
    - Summaries should prioritise what changed, why it matters, and recommended follow-ups.

    TONE:
    Speak like an operations / CX owner. Be factual, concise, and action-focused.
    """
    user_query: str = dspy.InputField(desc="The user query")
    filters: Optional[List[Dict[str, Any]]] = dspy.InputField(desc= "List of filters created so far in the conversation")
    user_history: List[str] = dspy.InputField(desc="Previous user messages in the conversation")
    assistant_history: List[str] = dspy.InputField(desc="Previous assistant messages in the conversation")
    final_answer: str = dspy.OutputField(desc="The final answer to the user query")

main_agent = dspy.ReAct(
    MainReactAgent,
    tools=[
        plot_chart.make_chart_spec,
        resolve_filter.create_filter_record,
        resolve_filter.update_filter_record,
        resolve_filter.resolve_filters,
        trend_breakdown.get_trend_breakdown,
        calculate_metrics.calculate_metrics,
        get_examples.get_examples,
    ],
    max_iters=30,
)

class DashboardStatusMessageProvider(StatusMessageProvider):
    @staticmethod
    def _sanitize(value):
        import pandas as _pd  # local import to avoid circularities

        if isinstance(value, _pd.DataFrame):
            return {
                "type": "DataFrame",
                "rows": len(value),
                "columns": list(value.columns),
            }
        if isinstance(value, dict):
            return {k: DashboardStatusMessageProvider._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [DashboardStatusMessageProvider._sanitize(v) for v in value]
        return value

    @staticmethod
    def _short_json(obj: Optional[object], limit: int = 200) -> str:
        if obj is None:
            return ""
        obj = DashboardStatusMessageProvider._sanitize(obj)
        try:
            text = json.dumps(obj, default=str)
        except TypeError:
            text = str(obj)
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text

    def lm_start_status_message(self, instance, inputs):
        return "Agent is reasoning..."

    def lm_end_status_message(self, outputs):
        return "Received model response."

    def tool_start_status_message(self, instance, inputs):
        name = getattr(instance, "name", getattr(instance, "__name__", "tool"))
        return f"Calling tool `{name}` with {self._short_json(inputs)}"

    def tool_end_status_message(self, outputs):
        return f"Tool finished. Output: {self._short_json(outputs)}"


_STREAM_LISTENERS = [
    StreamListener(signature_field_name="next_thought", allow_reuse=True),
    StreamListener(signature_field_name="final_answer"),
]

stream_main_agent = dspy.streamify(
    main_agent,
    stream_listeners=_STREAM_LISTENERS,
    status_message_provider=DashboardStatusMessageProvider(),
    async_streaming=False,
)

THEME_COLORS = {
    "light": {
        "primary_text": "#1F2A44",
        "secondary_text": "#5C6C83",
        "accent_color": "#FF7633",
        "surface": "#F6F8FC",
        "card_surface": "#FFFFFF",
        "card_border": "#E7ECF3",
        "card_shadow": "0px 4px 12px rgba(31, 42, 68, 0.06)",
        "divider": "rgba(31, 42, 68, 0.1)",
    },
    "dark": {
        "primary_text": "#F3F4F6",
        "secondary_text": "#9CA3AF",
        "accent_color": "#FF8B4D",
        "surface": "#0F172A",
        "card_surface": "#1E293B",
        "card_border": "#26334C",
        "card_shadow": "0px 12px 24px rgba(3, 7, 18, 0.55)",
        "divider": "rgba(148, 163, 184, 0.25)",
    },
}

def convert_filtered_df_to_excel(df):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Filtered_Tickets")

        # Access workbook + sheet
        ws = writer.book["Filtered_Tickets"]

        # Auto-adjust column width (important for usability)
        for column_cells in ws.columns:
            length = max(len(str(cell.value)) if cell.value else 0 for cell in column_cells)
            ws.column_dimensions[column_cells[0].column_letter].width = min(length + 3, 40)

        # Freeze header row
        ws.freeze_panes = "A2"

        # Turn on filters
        ws.auto_filter.ref = ws.dimensions

    return output.getvalue()


def detect_base_theme() -> str:
    """Return 'dark' or 'light' depending on the current browser theme."""
    if not st_theme:
        return "light"
    try:
        theme_info = st_theme()
    except Exception:
        return "light"
    base = theme_info.get("base") if isinstance(theme_info, dict) else None
    return "dark" if base and base.lower().startswith("dark") else "light"


def build_page_style(palette: dict, base: str) -> str:
    """Generate CSS using the palette detected from the browser theme."""
    return f"""
<style>
    :root {{
        color-scheme: {base};
        --primary-text: {palette["primary_text"]};
        --secondary-text: {palette["secondary_text"]};
        --accent-color: {palette["accent_color"]};
        --surface-color: {palette["surface"]};
        --card-surface: {palette["card_surface"]};
        --card-border: {palette["card_border"]};
        --card-shadow: {palette["card_shadow"]};
        --divider-color: {palette["divider"]};
    }}
    body {{
        background-color: var(--surface-color);
    }}
    .main .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
        background-color: var(--surface-color);
        transition: background-color 0.3s ease;
    }}
    .dashboard-title {{
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--primary-text);
        margin-bottom: 0.5rem;
    }}
    .dashboard-subtitle {{
        color: var(--secondary-text);
        font-size: 1rem;
        margin-bottom: 2rem;
    }}
    .section-divider {{
        margin: 2.25rem 0 1.75rem 0;
        border: none;
        border-top: 1px solid var(--divider-color);
    }}
    .section-heading {{
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--primary-text);
        margin-bottom: 0.35rem;
    }}
    .section-caption {{
        font-size: 0.9rem;
        color: var(--secondary-text);
        margin-bottom: 1rem;
    }}
    .kpi-card {{
        background-color: var(--card-surface);
        padding: 1rem 1.2rem;
        border-radius: 0.9rem;
        border: 1px solid var(--card-border);
        box-shadow: var(--card-shadow);
        min-height: 120px;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }}
    .kpi-label {{
        display: block;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--secondary-text);
    }}
    .kpi-value {{
        display: block;
        margin-top: 0.5rem;
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--primary-text);
    }}
    .filter-grid {{
        background-color: var(--card-surface);
        padding: 1.5rem 1.75rem;
        border-radius: 1rem;
        border: 1px solid var(--card-border);
        box-shadow: var(--card-shadow);
    }}
    .stMultiSelect label, .stDateInput label {{
        font-weight: 600 !important;
        color: var(--primary-text) !important;
    }}
    .stMultiSelect div[data-baseweb="select"],
    .stDateInput div[data-baseweb="input"] {{
        border-radius: 0.6rem;
        background-color: var(--card-surface);
        border: 1px solid var(--card-border);
        color: var(--primary-text);
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }}
    .stMultiSelect div[data-baseweb="select"] span,
    .stDateInput div[data-baseweb="input"] span {{
        color: var(--primary-text);
    }}
    .stDataFrame {{
        border-radius: 0.9rem;
        overflow: hidden;
        border: 1px solid var(--card-border);
    }}
    .stButton button {{
        background: var(--accent-color);
        color: #FFFFFF;
        border: none;
        border-radius: 0.6rem;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        transition: filter 0.2s ease, transform 0.2s ease;
    }}
    .stButton button:hover {{
        filter: brightness(1.05);
        transform: translateY(-1px);
    }}
</style>
"""

COLUMN_NAMES = [
    "complaint_number",
    "city",
    "region",
    "created_date",
    "refund_count_in_15_days",
    "product",
    "concern_type",
    "level_1_classification",
    "level_2_classification",
    "expanded_description",
    "customer_issue",
    "root_cause",
    "resolution_provided_summary",
    "product_category",
]

# check if vector_db folder exists, if not create it
if not os.path.exists("VectorDB"):
    os.makedirs("VectorDB")
# Download Vector DB files
today_date = datetime.date.today()

    
if 'summary_output' not in st.session_state:
    st.session_state.summary_output = ""
if 'tickets' not in st.session_state:
    st.session_state.tickets = []
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = ""
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = pd.DataFrame()
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'vector_db_initialized' not in st.session_state:
    st.session_state.vector_db_initialized = True
if 'user_messages' not in st.session_state:
    st.session_state.user_messages = []
if 'assistant_messages' not in st.session_state:
    st.session_state.assistant_messages = []
if 'chart_spec' not in st.session_state:
    st.session_state.chart_spec = []
if 'agent_last_step_trace' not in st.session_state:
    st.session_state.agent_last_step_trace = ""
if 'full_query_text' not in st.session_state:
    st.session_state.full_query_text = ""
if 'agent_tool_events' not in st.session_state:
    st.session_state.agent_tool_events = []
if 'filters_created' not in st.session_state:
    st.session_state.filters_created = []
cerebras_key = st.secrets.get("CEREBRAS_API_KEY")
if not cerebras_key:
    st.session_state.summary_output = "API key not configured. Check Streamlit Cloud secrets."
    
def summarise_ticket():
    tickets = st.session_state.tickets[:100]
    try:
        summary = ticket_summariser.summarise_texts(tickets)
    except ValueError:
        st.session_state.summary_output = "No tickets selected for summary."
    except Exception as exc:
        st.session_state.summary_output = f"Error generating summary: {exc}"
    else:
        st.session_state.summary_output = summary

    

@st.cache_data(show_spinner=False)
def load_ticket_dataframe(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df = df[df["CONCERN AREA NAME"] != "Stop Customer"]
    df = df[df["CONCERN TYPE NAME"] != "Internal"]
    categories = product_category.categories
    product_to_category = {product: cat for cat, products in categories.items() for product in products}
    df["product_category"] = df["product"].map(product_to_category).fillna("")
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_").lower())
    df.columns = [c.strip() for c in df.columns]
    df = df[COLUMN_NAMES]
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce").dt.date
    return df


def apply_selected_filters(df: pd.DataFrame, selections: dict, exclude: Optional[str] = None) -> pd.DataFrame:
    filtered = df
    date_range = selections.get("created_date")
    if exclude != "created_date" and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            filtered = filtered[
                (filtered["created_date"] >= start_date) & (filtered["created_date"] <= end_date)
            ]
    for col, selected_values in selections.items():
        if col == "created_date" or col == exclude or not selected_values:
            continue
        filtered = filtered[filtered[col].isin(selected_values)]
    return filtered


def get_available_options(df: pd.DataFrame, selections: dict, target_col: str) -> List:
    subset = apply_selected_filters(df, selections, exclude=target_col)
    return sorted(subset[target_col].dropna().unique().tolist())

def apply_text_filters(df: pd.DataFrame, full_query: str = "") -> pd.DataFrame:
    # AND across words within each field. Case-insensitive. Ignores NaNs.
    out = df
    if full_query:
        words = full_query.split()
        mask = pd.Series(True, index=out.index)
        for w in words:
            mask &= out["expanded_description"].astype(str).str.contains(w, case=False, na=False)
        out = out[mask]
    return out

def ensure_multiselect_state(key: str, options: List) -> None:
    current = st.session_state.get(key, [])
    if not isinstance(current, list):
        current = [current]
    valid = [value for value in current if value in options]
    if key not in st.session_state or len(valid) != len(current):
        st.session_state[key] = valid


def render_conversation_for(prefix, placeholder):
    try:
        placeholder.empty()
        with placeholder.container():
            user_history = st.session_state.get(f"{prefix}_user_messages", [])
            assistant_history = st.session_state.get(f"{prefix}_assistant_messages", [])

            if user_history or assistant_history:
                for user_msg, assistant_msg in zip_longest(user_history, assistant_history):
                    if user_msg:
                        with st.chat_message("user"):
                            st.markdown(user_msg)

                    if assistant_msg:
                        with st.chat_message("assistant"):
                            # This preserves JSON parsing, charts, metrics, examples etc
                            render_assistant_message(assistant_msg)

            # === Parity: Agent reasoning trace ===
            last_trace = st.session_state.get(f"{prefix}_agent_last_step_trace")
            if last_trace:
                with st.expander("Latest agent steps"):
                    st.markdown(last_trace)

            # === Parity: Tool call trace ===
            tool_events = st.session_state.get(f"{prefix}_agent_tool_events", [])
            if tool_events:
                with st.expander("Tool call trace"):
                    st.markdown("\n".join(f"- {event}" for event in tool_events))

    except Exception as exc:
        # Same crash-protection behavior as original
        st.error("Failed to render chat")
        st.write(exc)


def _parse_assistant_payload(final_answer: Any) -> Optional[Dict[str, Any]]:
    if isinstance(final_answer, dict):
        return final_answer
    if isinstance(final_answer, str):
        candidate = final_answer.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None
    return None


def _coerce_chart_candidate(candidate: Any) -> Optional[Dict[str, Any]]:
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        return plot_chart.normalize_chart_state(candidate)
    if isinstance(candidate, str):
        text = candidate.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return plot_chart.normalize_chart_state(parsed)
    return None


def _normalize_chart_list(candidate: Any) -> List[Dict[str, Any]]:
    if candidate is None:
        return []
    items = candidate if isinstance(candidate, list) else [candidate]
    specs: List[Dict[str, Any]] = []
    for item in items:
        normalized = _coerce_chart_candidate(item)
        if normalized:
            specs.append(normalized)
    return specs


def _collect_chart_specs_from_streams(streams: Dict[Tuple[str, str], str]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for (predict_name, _signature_field), raw in streams.items():
        if not predict_name or "make_chart_spec" not in predict_name:
            continue
        specs.extend(_normalize_chart_list(raw))
    return specs


def render_assistant_message(message: Any) -> None:
    charts: List[Dict[str, Any]] = []
    payload = message

    if isinstance(message, dict) and "content" in message and "chart_specs" in message:
        payload = message.get("content")
        charts = list(message.get("chart_specs") or [])

    parsed = _parse_assistant_payload(payload)
    if not parsed:
        st.markdown(payload)
    else:
        summary = parsed.get("summary")
        if summary:
            st.write(summary)

        metrics = parsed.get("metrics")
        if metrics:
            st.markdown("**Metrics**")
            st.json(metrics)

        examples = parsed.get("examples") or []
        if examples:
            st.markdown("**Examples**")
            st.json(examples)

        chart_state = parsed.get("chart")
        charts.extend(_normalize_chart_list(chart_state))

    if charts:
        st.markdown("**Chart**" if len(charts) == 1 else "**Charts**")
        for spec in charts:
            plot_chart.render_chart(spec)


def render_section_header(title: str, caption: Optional[str] = None) -> None:
    st.markdown(f"<div class='section-heading'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<div class='section-caption'>{caption}</div>", unsafe_allow_html=True)


def render_section_divider() -> None:
    st.markdown("<hr class='section-divider' />", unsafe_allow_html=True)


def render_kpi_cards(df: pd.DataFrame, title: str, caption: Optional[str] = None) -> None:
    render_section_header(title, caption)
    if df.empty:
        st.info("No tickets available for this view yet.")
        return
    metrics = [
        ("Total Tickets over last 15 days", len(df)),
        ("Unique Products", df["product"].nunique()),
        ("Concern Types", df["concern_type"].nunique()),
        ("Cities", df["city"].nunique()),
    ]
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.markdown(
            f"""
            <div class="kpi-card">
                <span class="kpi-label">{label}</span>
                <span class="kpi-value">{value:,}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )



def _google_api_key():
    return st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

def _drive_list(folder_id: str, api_key: str):
    url = (
        "https://www.googleapis.com/drive/v3/files"
        f"?q='{folder_id}'+in+parents"
        "&fields=files(id,name,mimeType)"
        f"&key={api_key}"
    )
    r = requests.get(url, timeout=30); r.raise_for_status()
    return r.json().get("files", []) or []

def _drive_get_shortcut_target(file_id: str, api_key: str):
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=id,name,mimeType,shortcutDetails(targetId,targetMimeType)&key={api_key}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    j = r.json()
    sd = j.get("shortcutDetails")
    if sd and sd.get("targetId"):
        return sd["targetId"], j.get("name"), sd.get("targetMimeType")
    return file_id, j.get("name"), j.get("mimeType")

def _drive_download(file_id: str, out_path: str, api_key: str):
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 256):
                if chunk:
                    f.write(chunk)
    return out_path

def run_agent_for(prefix, allowed_df=None, stream_area=None, conversation_placeholder=None, live_updates_placeholder=None):
    query = st.session_state.get(f"{prefix}_full_query_text", "").strip()
    if not query:
        return
    st.session_state[f"{prefix}_full_query_text"] = ""

    # Bind tools to filtered dataframe if provided
    if allowed_df is not None:
        trend_breakdown.register_ticket_dataframe(allowed_df)

    user_msgs = st.session_state[f"{prefix}_user_messages"]
    assistant_msgs = st.session_state[f"{prefix}_assistant_messages"]

    prior_user_history = user_msgs.copy()
    prior_assistant_history = [
        msg.get("content") if isinstance(msg, dict) and "content" in msg else msg
        for msg in assistant_msgs
    ]

    status_placeholder = stream_area.empty() if stream_area else st.empty()
    steps_placeholder = stream_area.empty() if stream_area else st.empty()

    aggregated_streams = defaultdict(str)
    captured_chart_specs = []
    final_prediction = None
    status_updates = []
    assistant_stub_index = None
    assistant_stream_placeholder = None

    user_msgs.append(query)
    with st.chat_message("user"):
        st.markdown(query)

    plot_chart.drain_chart_buffer()

    assistant_stub = {"content": "", "chart_specs": []}
    assistant_msgs.append(assistant_stub)
    assistant_stub_index = len(assistant_msgs) - 1

    with st.chat_message("assistant"):
        assistant_stream_placeholder = st.empty()
        assistant_stream_placeholder.markdown("...")

    try:
        with dspy.context(
            lm=dspy.LM(
                model="openai/gpt-oss-120b",
                api_key=cerebras_key,
                temperature=1,
                max_tokens=32000,
                api_base="https://api.cerebras.ai/v1",
            )
        ):
            for chunk in stream_main_agent(
                user_query=query,
                user_history=prior_user_history,
                assistant_history=prior_assistant_history,
                filters=st.session_state[f"{prefix}_filters_created"],
            ):
                if isinstance(chunk, StatusMessage):
                    status_updates.append(chunk.message)
                    if status_placeholder:
                        status_placeholder.markdown(
                            "**Status Updates**\n\n" + "\n".join(f"- {m}" for m in status_updates)
                        )
                    continue

                if isinstance(chunk, StreamResponse):
                    key = (chunk.predict_name, chunk.signature_field_name)
                    aggregated_streams[key] += chunk.chunk

                    if chunk.signature_field_name == "next_thought" and steps_placeholder:
                        steps_placeholder.markdown(
                            f"**Agent Steps**\n\n{aggregated_streams[key]}"
                        )
                    elif chunk.signature_field_name == "final_answer":
                        assistant_stream_placeholder.markdown(aggregated_streams[key] or "...")
                    continue

                if isinstance(chunk, dspy.Prediction):
                    final_prediction = chunk

        captured_chart_specs.extend(_collect_chart_specs_from_streams(aggregated_streams))

        if status_placeholder:
            status_placeholder.success("Response ready.")

    except Exception as exc:
        if status_placeholder:
            status_placeholder.error(f"Agent run failed: {exc}")

        if user_msgs:
            user_msgs.pop()
        if assistant_stub_index is not None and assistant_msgs:
            assistant_msgs.pop()

        if assistant_stream_placeholder:
            assistant_stream_placeholder.error("Agent run failed. Please try again.")

        st.session_state[f"{prefix}_agent_tool_events"] = status_updates

        if conversation_placeholder:
            render_conversation_for(prefix, conversation_placeholder)
        return

    if final_prediction is None:
        if user_msgs:
            user_msgs.pop()
        if assistant_stub_index is not None and assistant_msgs:
            assistant_msgs.pop()

        if status_placeholder:
            status_placeholder.error("Agent did not return a response.")
        if assistant_stream_placeholder:
            assistant_stream_placeholder.warning("Agent did not return a response.")

        st.session_state[f"{prefix}_agent_tool_events"] = status_updates

        if conversation_placeholder:
            render_conversation_for(prefix, conversation_placeholder)
        return

    final_answer = getattr(final_prediction, "final_answer", None)
    if final_answer is None:
        final_answer = aggregated_streams.get(("self", "final_answer"), "").strip()
    if not final_answer:
        final_answer = "No answer returned."

    parsed_answer = _parse_assistant_payload(final_answer)
    display_payload = parsed_answer if parsed_answer is not None else final_answer

    buffered_specs = _normalize_chart_list(plot_chart.drain_chart_buffer())
    existing_specs = _normalize_chart_list(st.session_state.get(f"{prefix}_chart_spec", []))
    final_chart_specs = []
    _seen_chart_specs = set()

    def _add_specs(specs):
        for spec in specs:
            normalized = plot_chart.normalize_chart_state(spec)
            if not normalized:
                continue
            key = json.dumps(normalized, sort_keys=True)
            if key not in _seen_chart_specs:
                _seen_chart_specs.add(key)
                final_chart_specs.append(normalized)

    _add_specs(buffered_specs)
    _add_specs(existing_specs)
    _add_specs(captured_chart_specs)

    chart_attr = getattr(final_prediction, "chart", None)
    _add_specs(_normalize_chart_list(chart_attr))
    if parsed_answer:
        _add_specs(_normalize_chart_list(parsed_answer.get("chart")))

    st.session_state[f"{prefix}_chart_spec"] = final_chart_specs

    assistant_record = {
        "content": display_payload,
        "chart_specs": final_chart_specs,
    }

    assistant_msgs[assistant_stub_index] = assistant_record

    if assistant_stream_placeholder:
        if parsed_answer and parsed_answer.get("summary"):
            assistant_stream_placeholder.markdown(parsed_answer["summary"])
        elif isinstance(final_answer, str):
            assistant_stream_placeholder.markdown(final_answer)
        else:
            assistant_stream_placeholder.markdown(json.dumps(final_answer, default=str))

    st.session_state[f"{prefix}_agent_last_step_trace"] = aggregated_streams.get(("self", "next_thought"), "")
    st.session_state[f"{prefix}_agent_tool_events"] = status_updates

    if steps_placeholder:
        steps_placeholder.markdown(
            f"**Agent Steps**\n\n{st.session_state[f'{prefix}_agent_last_step_trace'] or 'No steps captured.'}"
        )

    if conversation_placeholder:
        render_conversation_for(prefix, conversation_placeholder)

        
        
        
if __name__ == "__main__":

    base_theme = detect_base_theme()
    palette = THEME_COLORS.get(base_theme, THEME_COLORS["light"])
    st.set_page_config(
    page_title="Customer Support Insights",
    page_icon="🧊",
    layout="wide",
    menu_items={
        'Report a bug': "mailto:lakshaydagar@countrydelight.in",
        'About': "This app is built using Streamlit and DSPy to provide insights into customer support tickets."
    }, 
    initial_sidebar_state="expanded",
    )
    st.markdown(build_page_style(palette, base_theme), unsafe_allow_html=True)
    logo_path = "Frame 6.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=220)
    st.markdown(
        "<div class='dashboard-title'>Customer Support Ticket Analytics - Experimental</div>",
        unsafe_allow_html=True,
    )
    st.markdown("##### AI-powered analytics over customer support data")

    tab_agent, tab_table, tab_about = st.tabs(["# AI Assistant", "# Ticket Dashboard", "# About"])
    with tab_agent:
        conversation_placeholder = st.empty()
        render_conversation_for("global", conversation_placeholder)


        live_updates_placeholder = st.empty()
        stream_area = st.container()
        prompt = st.chat_input("Ask anything about the tickets", key="chat_prompt")
        if prompt:
            st.session_state.full_query_text = prompt
            run_agent_for("global", stream_area, conversation_placeholder, live_updates_placeholder)
    with tab_table:

        st.session_state.df = load_ticket_dataframe(DATA_URL)
        df = st.session_state.df
        catalog_loader.sync_filter_catalogs(df)
        trend_breakdown.register_ticket_dataframe(df)
        if not st.session_state.vector_db_initialized:
            latest_date_in_data = df["created_date"].max()
            if pd.notna(latest_date_in_data) and today_date > latest_date_in_data:
                api_key = _google_api_key()
                
                if not api_key:
                    st.error("Missing GOOGLE_API_KEY for Google Drive access.")
                else:
                    files = _drive_list(FOLDER_ID, api_key)
                    if len(files) != 1:
                        st.error(f"Expected exactly 1 file in folder, found {len(files)}.")
                    else:
                        fid, fname, mtype = files[0]["id"], files[0]["name"], files[0]["mimeType"]
                        # Resolve Google Drive shortcut if needed
                        if mtype == "application/vnd.google-apps.shortcut":
                            fid, fname, mtype = _drive_get_shortcut_target(fid, api_key)

                        # Enforce ZIP
                        if not (fname.lower().endswith(".zip") or mtype == "application/zip"):
                            st.error(f"File is not a ZIP (name={fname}, mimeType={mtype}).")
                        else:
                            zip_path = _drive_download(fid, os.path.join("downloads", fname), api_key)
                            with zipfile.ZipFile(zip_path, "r") as zf:
                                zf.extractall("VectorDB")
                            st.success(f"Downloaded and extracted: {fname}")

            st.session_state.vector_db_initialized = True
        st.markdown("You can use this to manually explore the tickets and summarise them, while visualising basic overall data distribution")
        with st.expander("Overall Ticket Distribution"):
            render_kpi_cards(
            df,
            "Tickets At A Glance",
            "Aggregated counts across the full dataset.",
            )
            render_section_divider()
            render_section_header(
                "Overall Ticket Distribution",
                "Understand how tickets break down before applying any filters.",
            )
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.markdown("**Concern Type Mix**")
                concern_counts = df['concern_type'].value_counts().reset_index()
                concern_counts.columns = ['Concern Type', 'Count']
                fig_concern_type = px.pie(concern_counts, values='Count', names='Concern Type')
                st.plotly_chart(fig_concern_type)
            with col_2:
                st.markdown("**Tickets by Region**")
                region_counts = df['region'].value_counts().reset_index()
                region_counts.columns = ['Region', 'Count']
                vc = df['region'].value_counts(dropna=False).rename_axis('Region').reset_index(name='Count')

                total = vc['Count'].sum()
                thresh = 0.01 * total  # 1%
                small = vc['Count'] < thresh
                region_counts = pd.concat(
                    [vc[~small], pd.DataFrame([{'Region':'Other','Count': vc.loc[small,'Count'].sum()}])]
                    , ignore_index=True
                ) if small.any() else vc
                fig_region = px.pie(region_counts, values='Count', names='Region')
                st.plotly_chart(fig_region)
            with col_3:
                st.markdown("**Level 1 Categories**")
                level1_counts = df['level_1_classification'].value_counts().reset_index()
                level1_counts.columns = ['Level 1 Category', 'Count']
                fig_level1 = px.pie(level1_counts, values='Count', names='Level 1 Category')
                st.plotly_chart(fig_level1)

            col_4, col_5, col_6 = st.columns(3)
            with col_4:
                st.markdown("**Level 2 Focus Areas**")
                level2_counts = df['level_2_classification'].value_counts().head(10)
                st.bar_chart(level2_counts, horizontal=True, x_label='Count', y_label='Issue', width = 'content', sort = '-count', color = "#FF7633")
            with col_5:
                st.markdown("**Top Products**")
                product_counts = df['product'].value_counts().head(10)
                st.bar_chart(product_counts, horizontal=True, x_label='Count', y_label='Product', width = 'content', sort = '-count', color = "#337EFF")
            with col_6:
                st.markdown("**Tickets by City**")
                city_counts = df['city'].value_counts().head(10)
                st.bar_chart(city_counts, horizontal=True, x_label='Count', y_label='City', width = 'content', sort = '-count', color = "#FFEB33")


        filter_config = [
            ("concern_type", "Concern Type", "filter_concern_type"),
            ("region", "Region", "filter_region"),
            ("level_1_classification", "Level 1 Classification", "filter_level_1"),
            ("level_2_classification", "Level 2 Classification", "filter_level_2"),
            ("product", "Product", "filter_product"),
            ("city", "City", "filter_city"),
        ]
        st.markdown("### Filters")
        st.markdown(
            """Use the controls below to filter tickets based on various attributes.
            """
        )
        min_val = df['created_date'].min()
        max_val = df['created_date'].max()
        if pd.isna(min_val) or pd.isna(max_val):
            min_val = max_val = today_date
        date_range_selection = st.date_input(
                label="Date Range",
                value=(min_val, max_val),
                min_value=min_val,
                max_value=max_val,
                key="filter_created_date",
            )
        if isinstance(date_range_selection, (list, tuple)) and len(date_range_selection) == 2:
            start_date, end_date = date_range_selection
        else:
            start_date = end_date = date_range_selection
        selected_filters = {"created_date": (start_date, end_date)}
        for column_name, label, key in filter_config:
            options = get_available_options(df, selected_filters, column_name)
            ensure_multiselect_state(key, options)
            selected_filters[column_name] = st.multiselect(label, options, key=key)
        product_categories = sorted(df['product_category'].dropna().unique().tolist())
        ensure_multiselect_state("filter_product_category", product_categories)
        selected_filters["product_category"] = st.multiselect("Product Category", product_categories, key="filter_product_category")
            

        st.session_state.filtered_df = apply_selected_filters(df, selected_filters).copy()
        filtered_df = st.session_state.filtered_df
        
        with st.expander("Filtered Ticket Snapshot and Distribution"):
            render_section_divider()
            render_kpi_cards(
                filtered_df,
                "Filtered Ticket Snapshot",
                "Metrics based on the active filters above.",
            )
            st.markdown("##")
            render_section_header(
                "Filtered Ticket Distribution"
            )

            if filtered_df.empty:
                st.info("No charts to display for the current filters. Adjust selections above to explore more data.")
            else:
                col_21, col_22, col_23 = st.columns(3)
                with col_21:
                    st.markdown("**Concern Type Mix (Filtered)**")
                    concern_filtered = filtered_df['concern_type'].value_counts().reset_index()
                    concern_filtered.columns = ['Concern Type', 'Count']
                    fig_concern_type_filter = px.pie(concern_filtered, values='Count', names='Concern Type')
                    st.plotly_chart(fig_concern_type_filter, key="pie_concern_type_filtered")
                with col_22:
                    st.markdown("**Tickets by Region (Filtered)**")
                    region_filtered = filtered_df['region'].value_counts().reset_index()
                    region_filtered.columns = ['Region', 'Count']
                    vc = filtered_df['region'].value_counts(dropna=False).rename_axis('Region').reset_index(name='Count')
                    total = vc['Count'].sum()
                    thresh = 0.01 * total  # 1%
                    small = vc['Count'] < thresh
                    region_counts = pd.concat(
                        [vc[~small], pd.DataFrame([{'Region':'Other','Count': vc.loc[small,'Count'].sum()}])]
                        , ignore_index=True
                    ) if small.any() else vc
                    fig_region_filter = px.pie(region_counts, values='Count', names='Region')
                    st.plotly_chart(fig_region_filter, key="pie_region_filtered")
                with col_23:
                    st.markdown("**Level 1 Categories (Filtered)**")
                    level1_filtered = filtered_df['level_1_classification'].value_counts().reset_index()
                    level1_filtered.columns = ['Level 1 Category', 'Count']
                    fig_level1_filter = px.pie(level1_filtered, values='Count', names='Level 1 Category')
                    st.plotly_chart(fig_level1_filter, key="pie_level1_filtered")

                col_24, col_25, col_26 = st.columns(3)
                with col_24:
                    st.markdown("**Level 2 Focus Areas (Filtered)**")
                    level2_filtered = filtered_df['level_2_classification'].value_counts().head(10)
                    st.bar_chart(level2_filtered, horizontal=True, x_label='Count', y_label='Issue', width = 'content', sort='-count', color = "#FFCF33")
                with col_25:
                    st.markdown("**Top Products (Filtered)**")
                    product_filtered = filtered_df['product'].value_counts().head(10)
                    st.bar_chart(product_filtered, horizontal=True, x_label='Count', y_label='Product', width = 'content', sort='-count', color = "#FF337A")
                with col_26:
                    st.markdown("**Tickets by City (Filtered)**")
                    city_filtered = filtered_df['city'].value_counts().head(10)
                    st.bar_chart(city_filtered, horizontal=True, x_label='Count', y_label='City', width = 'content', sort='-count', color = "#336DFF")

                

        filtered_count = len(filtered_df)
        if filtered_count:
            caption = f"{filtered_count:,} tickets match the current filters."
        else:
            caption = "No tickets match the current filters. Adjust your selections above to see results."  
        with st.expander("View Filtered Tickets"):
            st.markdown('### Ticket Details')
            st.write(caption)

            if filtered_df.empty:
                st.info("Try broadening the filters to explore more tickets.")
            else:
                st.dataframe(filtered_df, width = 'content')
            excel_data = convert_filtered_df_to_excel(filtered_df)

            st.download_button(
                label="Download Filtered Data",
                data=excel_data,
                file_name="filtered_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            fp = df_fingerprint(filtered_df)
            if st.session_state["filtered_vector_scope_hash"] != fp:
                init_agent_state("filtered")
                st.session_state["filtered_vector_scope_hash"] = fp

            st.markdown("### Ask AI about this filtered data")

            chat_box = st.empty()
            render_conversation_for("filtered", chat_box)

            prompt = st.chat_input("Ask about these tickets", key="filtered_chat")
            if prompt:
                st.session_state["filtered_full_query_text"] = prompt
                run_agent_for("filtered", allowed_df=filtered_df, conversation_placeholder=chat_box)


        with st.expander("Summarise Filtered Tickets"):
            st.markdown("### Summarise Filtered Tickets")
            st.write("Generate a concise summary of the filtered tickets, focusing on specific aspects if desired.")
            summary_config = {
                "Full Ticket Narrative": ("expanded_description", "Expanded_Description_Collection"),
                "Customer Issue": ("customer_issue", "Customer_Issue_Collection"),
                "Resolution Provided": ("resolution_provided_summary", "Resolution_Provided_Collection"),
                "Root Cause": ("root_cause", "Root_Cause_Collection"),
            }
            user_selection = st.segmented_control(
                label="Select narrative focus",
                options=list(summary_config.keys()),
                key="action_control",
                default= "Full Ticket Narrative",
            )
            column_name, collection_name = summary_config[user_selection]
            st.session_state.tickets = filtered_df[column_name].dropna().tolist()
            st.session_state.collection_name = collection_name
            st.button('Generate Summary', on_click=summarise_ticket, type="primary")
            st.markdown(st.session_state.summary_output or "*No summary generated yet.*")

    with tab_about:
        st.markdown("### About This Dashboard")
        st.write(
            """
            This app was built to analyse customer support tickets, helping teams identify key issues and areas for improvement.
            
            Currently the dashboard shows metrics from last 15 days of customer support tickets.
            We deliberately kept the some tickets out of the analysis to ensure that insights are drawn from relevant tickets only.
            The excluded tickets are which request for stopping the service ("because they subscribed and service was not available to their location") and internal tickets which are raised by the Customer Support Team to communicate among themselves.

            For any questions or feedback, please contact [Nishant Rajpoot](mailto:nishantrajpoot@countrydelight.in).
            """)
        st.divider()
        st.write("© 2025 Country Delight")
        st.write("Built with ❤️ by Digital Innovations Team | Country Delight")













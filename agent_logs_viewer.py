import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

LOG_PATH = Path(__file__).resolve().parent / "logs" / "agent_steps.log"

COLOR_MAP = {
    "thought": "#1f77b4",
    "tool_call": "#d62728",
    "tool_input": "#ff9896",
    "tool_output": "#98df8a",
    "status": "#2ca02c",
    "summary": "#9467bd",
    "trace": "#8c564b",
    "final_json": "#17becf",
    "final_text": "#ff7f0e",
}


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        ts = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None
    remainder = line[24:]
    if " " not in remainder:
        return None
    level, message = remainder.split(" ", 1)
    return {"timestamp": ts, "level": level, "message": message}


def load_runs() -> List[Dict[str, Any]]:
    if not LOG_PATH.exists():
        return []

    runs: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}

    with LOG_PATH.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            entry = parse_log_line(raw_line)
            if not entry:
                continue
            ts = entry["timestamp"]
            message = entry["message"]

            if message.startswith("QUERY:"):
                if current:
                    current["end"] = current.get("end", current["start"])
                    runs.append(current)
                current = {
                    "query": message.replace("QUERY:", "", 1).strip(),
                    "start": ts,
                    "end": ts,
                    "events": [],
                    "timeline": [],
                    "final_answer_json": None,
                    "final_answer_text": None,
                    "agent_trace": None,
                }

            if not current:
                continue

            current["end"] = ts
            current["events"].append(entry)

            if message.startswith("STATUS:"):
                detail = message.replace("STATUS:", "", 1).strip()
                entry_type = "tool_call" if "`" in message or "Calling tool" in detail else "status"
                current["timeline"].append(
                    {
                        "timestamp": ts,
                        "type": entry_type,
                        "text": detail,
                    }
                )
            elif message.startswith("TOOL INPUT"):
                detail = message.replace("TOOL INPUT", "", 1).strip()
                current["timeline"].append(
                    {
                        "timestamp": ts,
                        "type": "tool_input",
                        "text": detail,
                    }
                )
            elif message.startswith("TOOL OUTPUT"):
                detail = message.replace("TOOL OUTPUT", "", 1).strip()
                current["timeline"].append(
                    {
                        "timestamp": ts,
                        "type": "tool_output",
                        "text": detail,
                    }
                )
            elif message.startswith("STREAM "):
                _, remainder = message.split(" ", 1)
                if ": " in remainder:
                    stream_name, chunk = remainder.split(": ", 1)
                    current["timeline"].append(
                        {
                            "timestamp": ts,
                            "type": "thought_chunk",
                            "stream": stream_name,
                            "text": chunk,
                        }
                    )
            elif message.startswith("FINAL ANSWER JSON:"):
                payload = message.replace("FINAL ANSWER JSON:", "", 1).strip()
                try:
                    current["final_answer_json"] = json.loads(payload)
                except json.JSONDecodeError:
                    current["final_answer_text"] = payload
                current["timeline"].append(
                    {
                        "timestamp": ts,
                        "type": "final_json",
                        "text": payload,
                    }
                )
            elif message.startswith("FINAL ANSWER:"):
                detail = message.replace("FINAL ANSWER:", "", 1).strip()
                current["final_answer_text"] = detail
                current["timeline"].append(
                    {
                        "timestamp": ts,
                        "type": "final_text",
                        "text": detail,
                    }
                )
            elif message.startswith("AGENT STEPS TRACE:"):
                detail = message.replace("AGENT STEPS TRACE:", "", 1).strip()
                current["agent_trace"] = detail
                current["timeline"].append(
                    {
                        "timestamp": ts,
                        "type": "trace",
                        "text": detail,
                    }
                )
            elif message.startswith("STATUS SUMMARY:"):
                detail = message.replace("STATUS SUMMARY:", "", 1).strip()
                current["timeline"].append(
                    {
                        "timestamp": ts,
                        "type": "summary",
                        "text": detail,
                    }
                )

    if current:
        current["end"] = current.get("end", current["start"])
        runs.append(current)

    for run in runs:
        duration = (run["end"] - run["start"]).total_seconds()
        run["duration_seconds"] = duration

        consolidated: List[Dict[str, Any]] = []
        thought_buffer = ""
        thought_stream = ""
        thought_start = None

        def flush_thought():
            nonlocal thought_buffer, thought_stream, thought_start
            if thought_buffer:
                consolidated.append(
                    {
                        "timestamp": thought_start,
                        "type": "thought",
                        "stream": thought_stream,
                        "text": " ".join(thought_buffer.split()),
                    }
                )
                thought_buffer = ""
                thought_stream = ""
                thought_start = None

        for item in run.get("timeline", []):
            if item["type"] == "thought_chunk":
                chunk = item["text"]
                stream_name = item["stream"]
                if thought_stream != stream_name:
                    flush_thought()
                    thought_stream = stream_name
                    thought_start = item["timestamp"]
                thought_buffer += chunk
            else:
                flush_thought()
                consolidated.append(item)
        flush_thought()

        consolidated.sort(key=lambda x: x["timestamp"])
        run["timeline"] = consolidated

    return runs


def render_run(run: Dict[str, Any]) -> None:
    start = run["start"].strftime("%Y-%m-%d %H:%M:%S")
    duration = f"{run['duration_seconds']:.2f}s"
    with st.expander(f"{start} - {run['query']} (duration {duration})", expanded=False):
        st.markdown(f"**Query:** {run['query']}")
        st.markdown(f"**Start:** {start}")
        st.markdown(f"**Duration:** {duration}")

        st.markdown("**Conversation Timeline**")
        for item in run.get("timeline", []):
            label = item["timestamp"].strftime("%H:%M:%S")
            color = COLOR_MAP.get(item["type"], "#444444")

            if item["type"] == "thought":
                prefix = f"Thought ({item.get('stream')})"
                body = item["text"]
                html = (
                    f"<div style='color:{color}; margin-bottom:4px;'>"
                    f"<strong>[{label}] {prefix}:</strong> {body}"
                    "</div>"
                )
                st.markdown(html, unsafe_allow_html=True)
                continue

            if item["type"] in {"tool_call", "tool_input", "tool_output"}:
                prefix_map = {
                    "tool_call": "Tool Call",
                    "tool_input": "Tool Input",
                    "tool_output": "Tool Output",
                }
                prefix = prefix_map.get(item["type"], "Tool Event")
                raw_text = item.get("text", "")
                json_payload = None
                label_text = prefix

                if item["type"] in {"tool_input", "tool_output"} and ": " in raw_text:
                    name, payload_str = raw_text.split(": ", 1)
                    label_text = f"{prefix} ({name.strip()})"
                    try:
                        json_payload = json.loads(payload_str)
                    except json.JSONDecodeError:
                        json_payload = None
                        body = raw_text
                else:
                    body = raw_text

                if json_payload is not None:
                    st.markdown(
                        f"<div style='color:{color}; margin-bottom:4px;'><strong>[{label}] {label_text}:</strong></div>",
                        unsafe_allow_html=True,
                    )
                    st.json(json_payload)
                else:
                    html = (
                        f"<div style='color:{color}; margin-bottom:4px;'>"
                        f"<strong>[{label}] {label_text}:</strong> {body}"
                        "</div>"
                    )
                    st.markdown(html, unsafe_allow_html=True)
                continue

            if item["type"] == "status":
                prefix = "Status"
                body = item["text"]
            elif item["type"] == "summary":
                prefix = "Summary"
                body = item["text"]
            elif item["type"] == "trace":
                prefix = "Trace"
                body = item["text"]
            elif item["type"] == "final_json":
                prefix = "Final Answer (JSON)"
                body = "Captured structured answer."
            elif item["type"] == "final_text":
                prefix = "Final Answer"
                body = item["text"]
            else:
                prefix = item["type"].title()
                body = item.get("text", "")

            html = (
                f"<div style='color:{color}; margin-bottom:4px;'>"
                f"<strong>[{label}] {prefix}:</strong> {body}"
                "</div>"
            )
            st.markdown(html, unsafe_allow_html=True)

        if run.get("final_answer_json"):
            st.markdown("**Final Answer (JSON)**")
            st.json(run["final_answer_json"])
        elif run.get("final_answer_text"):
            st.markdown("**Final Answer**")
            st.write(run["final_answer_text"])

        if run["events"]:
            if st.checkbox(
                f"Show raw log lines ({run['start'].strftime('%H:%M:%S')})",
                key=f"raw_{run['start'].timestamp()}",
            ):
                st.code(
                    "\n".join(
                        f"{entry['timestamp'].strftime('%H:%M:%S')} {entry['level']} {entry['message']}"
                        for entry in run["events"]
                    ),
                    language="text",
                )


def main() -> None:
    st.set_page_config("Agent Log Viewer", layout="wide")
    st.title("Agent Log Viewer")
    st.write("Chronological view of agent queries, thoughts, tool calls, and outputs.")

    if not LOG_PATH.exists():
        st.info(f"No log file found at `{LOG_PATH}`")
        return

    runs = load_runs()
    if not runs:
        st.info("No runs captured yet.")
        return

    st.markdown(f"Loaded **{len(runs)}** runs from `{LOG_PATH}`")
    for run in reversed(runs):
        render_run(run)


if __name__ == "__main__":
    main()

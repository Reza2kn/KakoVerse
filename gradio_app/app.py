import json
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import gradio as gr
from datasets import Dataset, load_dataset

DEFAULT_DATASET_ID = "reza2kn/kakoverse-conversations-v0"


@lru_cache(maxsize=4)
def _load_dataset(repo_id: str) -> Dataset:
    return load_dataset(repo_id)["train"]


def load_conversations(repo_id: str) -> Tuple[List[Dict[str, Any]], List[List[Any]]]:
    dataset = _load_dataset(repo_id)
    table_rows: List[List[Any]] = []
    records: List[Dict[str, Any]] = []

    for row in dataset:
        records.append(row)
        table_rows.append(
            [
                row["persona_id"],
                row.get("crisis_category"),
                row.get("crisis_summary"),
                row.get("turns_recorded"),
                row.get("desired_turns"),
            ]
        )
    return records, table_rows


def conversation_to_html(conversation: Dict[str, Any]) -> str:
    turns = conversation.get("turns") or []
    persona_id = conversation.get("persona_id")
    crisis_category = conversation.get("crisis_category")
    crisis_summary = conversation.get("crisis_summary")

    head = f"""
    <style>
    .thread {{ font-family: Inter, sans-serif; max-width: 760px; margin: 1rem auto; }}
    .meta {{ margin-bottom: 1rem; color: #444; }}
    .bubble {{ border-radius: 14px; padding: 12px 16px; margin: 6px 0; display: inline-block; max-width: 90%; }}
    .seeker {{ background-color: #1f2937; color: white; float: left; clear: both; }}
    .supporter {{ background-color: #e5f3ff; color: #0f172a; float: right; clear: both; }}
    .cold {{ background-color: #f5f5f5; color: #1f2937; }}
    .care-pill {{ font-size: 12px; margin-left: 6px; padding: 2px 6px; border-radius: 999px; background: rgba(15, 23, 42, 0.1); }}
    .turn {{ overflow: hidden; padding-bottom: 12px; }}
    </style>
    <div class="thread">
    <div class="meta"><strong>Persona:</strong> {persona_id} | <strong>Crisis:</strong> {crisis_category} — {crisis_summary}</div>
    """

    body_parts = [head]
    for turn in turns:
        seeker_msg = turn.get("seeker_message", "")
        body_parts.append(f'<div class="turn"><div class="bubble seeker">{seeker_msg}</div></div>')
        for variant in turn.get("supporter_responses", []):
            care_level = variant.get("care_level")
            label = variant.get("style")
            classes = "bubble supporter"
            if label == "cold":
                classes += " cold"
            message_html = variant.get("message", "")
            body_parts.append(
                f'<div class="bubble {classes}">{message_html}'
                f'<span class="care-pill">{label} · care={care_level}</span></div>'
            )
    body_parts.append("</div>")
    return "".join(body_parts)


def on_load(repo_id: str):
    records, table_rows = load_conversations(repo_id)
    table_update = gr.update(value=table_rows, visible=True)
    status_update = gr.update(value=f"Loaded {len(records)} conversations from {repo_id}", visible=True)
    return records, table_update, status_update


def on_select(evt: gr.SelectData, records: List[Dict[str, Any]]) -> str:
    row_index = evt.index[0]
    if row_index < 0 or row_index >= len(records):
        return "<div>Select a valid conversation row.</div>"
    record = records[row_index]
    return conversation_to_html(record)


with gr.Blocks(css=".footer {display:none !important;}") as demo:
    gr.Markdown("# KakoVerse Conversations Browser")
    gr.Markdown(
        "Load a conversation dataset from the Hugging Face Hub, then double-click any row to view it in a chat-style layout."
    )

    dataset_state = gr.State([])
    with gr.Row():
        dataset_id = gr.Textbox(label="Dataset repo", value=DEFAULT_DATASET_ID, scale=4)
        browse_btn = gr.Button("Browse conversations", scale=1)
    status = gr.Markdown("", visible=False)

    table = gr.Dataframe(
        headers=["persona_id", "crisis_category", "crisis_summary", "turns_recorded", "desired_turns"],
        datatype=["str", "str", "str", "int", "int"],
        row_count=0,
        col_count=5,
        interactive=False,
        visible=False,
    )
    chat_panel = gr.HTML()

    browse_btn.click(
        on_load,
        inputs=[dataset_id],
        outputs=[dataset_state, table, status],
    )
    table.select(on_select, inputs=[dataset_state], outputs=chat_panel)


if __name__ == "__main__":
    demo.launch()

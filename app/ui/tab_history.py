"""
Gradio Tab 3 — Chat History.

Displays a log of all RAG queries made during the current session,
including the question, answer, sources, and parameters used.
Users can refresh or clear the history.
"""

import logging

import gradio as gr

from app.ui.tab_query import chat_history

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _format_history() -> str:
    """Render the chat history as formatted Markdown."""
    if not chat_history:
        return "📭 No queries yet. Go to the **Query** tab to ask a question."

    lines: list[str] = []

    for i, entry in enumerate(reversed(chat_history), start=1):
        sources_str = ", ".join(f"`{s}`" for s in entry["sources"]) if entry["sources"] else "None"
        params = entry["params"]

        lines.append(
            f"### Query {len(chat_history) - i + 1}\n"
            f"**🕐 {entry['timestamp']}**\n\n"
            f"**Question:** {entry['question']}\n\n"
            f"**Answer:**\n{entry['answer']}\n\n"
            f"**Sources:** {sources_str} · "
            f"*{entry['num_chunks']} chunk(s) retrieved*\n\n"
            f"<details><summary>Parameters</summary>\n\n"
            f"| Parameter | Value |\n"
            f"|-----------|-------|\n"
            f"| Top-K | {params['top_k']} |\n"
            f"| Search Type | {params['search_type']} |\n"
            f"| Max Tokens | {params['max_tokens']} |\n"
            f"| Temperature | {params['temperature']} |\n"
            f"| Top-P | {params['top_p']} |\n"
            f"| Repeat Penalty | {params['repeat_penalty']} |\n\n"
            f"</details>\n\n"
            f"---\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------


def handle_refresh() -> str:
    """Refresh the history display."""
    return _format_history()


def handle_clear() -> str:
    """Clear all chat history."""
    chat_history.clear()
    logger.info("Chat history cleared.")
    return "🗑️ History cleared."


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_tab() -> gr.TabItem:
    """Construct and return the Chat History tab."""

    with gr.TabItem("📜 History", id="tab_history") as tab:
        gr.Markdown("## Chat History\nReview past queries, answers, and the parameters used.")

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            clear_btn = gr.Button("🗑️ Clear History", variant="stop")

        history_display = gr.Markdown(
            value=_format_history(),
            label="Query History",
        )

        # --- Event wiring ---
        refresh_btn.click(
            fn=handle_refresh,
            inputs=[],
            outputs=[history_display],
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[],
            outputs=[history_display],
        )

    return tab

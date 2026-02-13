"""
Gradio Tab 2 — RAG Query.

Allows users to:
- Enter a question / prompt
- Customize the system prompt
- Adjust retrieval parameters (top-k, search type)
- Adjust generation parameters (max tokens, temperature, top-p, repeat penalty)
- Execute the RAG pipeline and see the answer with cited sources
"""

import logging
from datetime import datetime, timezone

import gradio as gr

from app.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_SEARCH_TYPE,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MAX_TOKENS_RANGE,
    REPEAT_PENALTY_RANGE,
    TEMPERATURE_RANGE,
    TOP_K_RANGE,
    TOP_P_RANGE,
)
from app.rag.chain import RAGRequest, RAGResponse, query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory chat history (shared with Tab 3)
# ---------------------------------------------------------------------------

# List of dicts: {"timestamp", "question", "answer", "sources", "params"}
chat_history: list[dict] = []


# ---------------------------------------------------------------------------
# Gradio event handler
# ---------------------------------------------------------------------------


def handle_query(
    question: str,
    system_prompt: str,
    top_k: int,
    search_type: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
) -> tuple[str, str]:
    """
    Execute a RAG query with user-specified parameters.

    Args:
        question:       The user's question.
        system_prompt:  System prompt for the LLM.
        top_k:          Number of chunks to retrieve.
        search_type:    "similarity" or "mmr".
        max_tokens:     Maximum tokens to generate.
        temperature:    Sampling temperature.
        top_p:          Nucleus sampling threshold.
        repeat_penalty: Repetition penalty.

    Returns:
        Tuple of (answer_display, sources_display).
    """
    if not question.strip():
        return "⚠️ Please enter a question.", ""

    request = RAGRequest(
        question=question.strip(),
        system_prompt=system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
        top_k=int(top_k),
        search_type=search_type,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        repeat_penalty=float(repeat_penalty),
    )

    logger.info("RAG query: %.100s...", request.question)

    response: RAGResponse = query(request)

    if response.error:
        return f"❌ {response.error}", ""

    # Format sources
    if response.sources:
        sources_text = "**Sources:** " + ", ".join(f"`{s}`" for s in response.sources)
        sources_text += f"\n\n*Retrieved {response.num_chunks_retrieved} chunk(s)*"
    else:
        sources_text = "*No sources retrieved.*"

    # Save to chat history for Tab 3
    chat_history.append({
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "question": request.question,
        "answer": response.answer,
        "sources": response.sources,
        "num_chunks": response.num_chunks_retrieved,
        "params": {
            "top_k": request.top_k,
            "search_type": request.search_type,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "repeat_penalty": request.repeat_penalty,
        },
    })

    return response.answer, sources_text


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_tab() -> gr.TabItem:
    """Construct and return the RAG Query tab."""

    with gr.TabItem("💬 Query", id="tab_query") as tab:
        gr.Markdown("## RAG Query\nAsk questions over your indexed documents.")

        # --- Question input ---
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What are the key findings in the Q3 report?",
            lines=3,
        )

        # --- System prompt (collapsible) ---
        with gr.Accordion("System Prompt", open=False):
            system_prompt_input = gr.Textbox(
                label="System Prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=4,
                info="Instructions that guide the LLM's behavior. The default works well for most RAG tasks.",
            )

        # --- Parameters (collapsible) ---
        with gr.Accordion("Retrieval Parameters", open=False):
            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=TOP_K_RANGE[0],
                    maximum=TOP_K_RANGE[1],
                    value=DEFAULT_TOP_K,
                    step=1,
                    label="Top-K Chunks",
                    info="Number of document chunks to retrieve. More chunks = more context but slower.",
                )
                search_type_dropdown = gr.Dropdown(
                    choices=["similarity", "mmr"],
                    value=DEFAULT_SEARCH_TYPE,
                    label="Search Type",
                    info="MMR (Maximal Marginal Relevance) adds diversity to retrieved chunks.",
                )

        with gr.Accordion("Generation Parameters", open=False):
            with gr.Row():
                max_tokens_slider = gr.Slider(
                    minimum=MAX_TOKENS_RANGE[0],
                    maximum=MAX_TOKENS_RANGE[1],
                    value=DEFAULT_MAX_TOKENS,
                    step=32,
                    label="Max Tokens",
                    info="Maximum number of tokens in the generated response.",
                )
                temperature_slider = gr.Slider(
                    minimum=TEMPERATURE_RANGE[0],
                    maximum=TEMPERATURE_RANGE[1],
                    value=DEFAULT_TEMPERATURE,
                    step=0.05,
                    label="Temperature",
                    info="Lower = more deterministic and factual. Higher = more creative.",
                )
            with gr.Row():
                top_p_slider = gr.Slider(
                    minimum=TOP_P_RANGE[0],
                    maximum=TOP_P_RANGE[1],
                    value=DEFAULT_TOP_P,
                    step=0.05,
                    label="Top-P (Nucleus Sampling)",
                    info="Cumulative probability threshold for token selection.",
                )
                repeat_penalty_slider = gr.Slider(
                    minimum=REPEAT_PENALTY_RANGE[0],
                    maximum=REPEAT_PENALTY_RANGE[1],
                    value=DEFAULT_REPEAT_PENALTY,
                    step=0.05,
                    label="Repeat Penalty",
                    info="Penalizes repeated tokens. Higher = less repetition.",
                )

        # --- Submit ---
        submit_btn = gr.Button("🚀 Generate Answer", variant="primary")

        # --- Output ---
        answer_output = gr.Textbox(
            label="Answer",
            lines=10,
            interactive=False,
            show_copy_button=True,
        )
        sources_output = gr.Markdown(value="", label="Sources")

        # --- Event wiring ---
        submit_btn.click(
            fn=handle_query,
            inputs=[
                question_input,
                system_prompt_input,
                top_k_slider,
                search_type_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                repeat_penalty_slider,
            ],
            outputs=[answer_output, sources_output],
        )

        # Also trigger on Enter key in question box
        question_input.submit(
            fn=handle_query,
            inputs=[
                question_input,
                system_prompt_input,
                top_k_slider,
                search_type_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                repeat_penalty_slider,
            ],
            outputs=[answer_output, sources_output],
        )

    return tab

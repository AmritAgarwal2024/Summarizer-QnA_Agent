# execution.py

import streamlit as st
from typing import Dict

from engine import (
    MODEL_CONFIG,
    initialize_models_and_chains,
    get_verified_response,
)
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from engine import ReconcilerOutput


# --------------------------
# Default settings
# --------------------------
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 1500
chunk_overlap = 100
k = 5


# --------------------------
# Prompt definitions
# --------------------------
qa_prompt = PromptTemplate.from_template(
    """
You are a financial assistant. Use the provided context to answer the question.

Context:
{context}

Question: {question}

Answer:
"""
)

# Parser for strict JSON reconciliation
parser = PydanticOutputParser(pydantic_object=ReconcilerOutput)

reconciler_prompt = PromptTemplate(
    template="""
You are reconciling multiple financial model answers.

Question: {question}
Context: {context}
All Answers: {all_answers}

Provide the best consolidated answer as JSON.
{format_instructions}
""",
    input_variables=["question", "context", "all_answers"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# --------------------------
# Streamlit UI
# --------------------------
st.title("📊 Financial Report Q&A Assistant")

# Upload section
uploaded_file = st.file_uploader("Upload a company's annual report (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    # Initialize retriever + models
    if "retriever" not in st.session_state:
        from engine import create_or_load_vectorstore

        retriever = create_or_load_vectorstore(
            uploaded_file.name,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
        )
        st.session_state.retriever = retriever

        # Initialize model chains
        chains, reconciler = initialize_models_and_chains(
            MODEL_CONFIG, st.session_state.retriever, qa_prompt, reconciler_prompt
        )
        st.session_state.chains = chains
        st.session_state.reconciler = reconciler


# --------------------------
# Q&A Section
# --------------------------
st.header("Ask a question")
question_input = st.text_input("Type your question here")
ask_btn = st.button("Ask")

if ask_btn and question_input:
    if not st.session_state.get("retriever"):
        st.warning("Please upload a PDF first.")
    else:
        # Get model responses + reconciliation
        final, per_model = get_verified_response(
            question_input,
            st.session_state.retriever,
            st.session_state.chains,
            st.session_state.reconciler,
        )

        try:
            parsed_result = parser.parse(final)
            final_answer = parsed_result.answer
            confidence = parsed_result.confidence
        except Exception as e:
            final_answer = final
            confidence = "N/A"
            st.error(f"Parser failed, showing raw output. Error: {e}")

        # Show result
        st.subheader("Answer")
        st.markdown(final_answer)
        st.caption(f"Confidence: {confidence}")

        # Expand for per-model answers
        with st.expander("See per-model answers"):
            for m, ma in per_model.items():
                st.markdown(f"**{m}** (confidence: {ma.confidence})\n\n{ma.answer}")

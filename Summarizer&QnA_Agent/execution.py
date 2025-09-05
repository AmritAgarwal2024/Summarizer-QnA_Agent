# app.py
import streamlit as st
import tempfile
from engine import (
    create_or_load_vectorstore,
    initialize_models_and_chains,
    get_verified_response,
    market_snapshot_md,
    EMBEDDING_MODEL_NAME_DEFAULT,
    ModelAnswer
)
from pathlib import Path
import os

st.set_page_config(page_title="Financial Report Q&A Assistant", layout="wide")
st.title("📊 Financial Report Q&A Assistant (Demo)")

# Sidebar controls
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type=["pdf"])
    ticker = st.text_input("Stock Ticker", value="RELIANCE.NS")
    st.markdown("---")
    st.subheader("Vectorstore / Embedding settings")
    embedding_model = st.selectbox("Embedding model", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"], index=0)
    chunk_size = st.number_input("Chunk size", min_value=512, max_value=4000, value=1500, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=800, value=100, step=50)
    k = st.number_input("Retriever k (per retriever)", min_value=1, max_value=20, value=5)
    st.markdown("---")
    st.subheader("Model toggles")
    use_gemini = st.checkbox("Gemini (Google)", value=True)
    use_deepseek = st.checkbox("DeepSeek (OpenRouter)", value=True)
    use_cohere = st.checkbox("Cohere", value=True)
    st.markdown("---")
    run_analysis_btn = st.button("📊 Generate Full Analysis")
    st.write("")
    st.subheader("Q&A")
    question_input = st.text_input("Ask a question from the uploaded report")
    ask_btn = st.button("💬 Ask")

# Session state for caching resources
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chains" not in st.session_state:
    st.session_state.chains = None
if "reconciler" not in st.session_state:
    st.session_state.reconciler = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# Handle upload => create/reuse vectorstore
if uploaded_file is not None and st.session_state.retriever is None:
    with st.spinner("Processing PDF & building vectorstore (this may take a moment)..."):
        # write temp file
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        t.write(uploaded_file.read())
        t.flush()
        t.close()
        st.session_state.pdf_path = t.name

        # create or load vectorstore
        retriever = create_or_load_vectorstore(
            t.name,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k
        )
        st.session_state.retriever = retriever
        st.success("✅ Vectorstore ready (cached by PDF hash).")

# Initialize models & chains (lazy)
if st.session_state.retriever and st.session_state.chains is None:
    with st.spinner("Initializing LLMs and RAG chains..."):
        # build a model_config subset based on toggles (re-use original MODEL_CONFIG like structure)
        MODEL_CONFIG = {}
        if use_gemini:
            MODEL_CONFIG["gemini"] = {
                "class": __import__("langchain_google_genai", fromlist=["GoogleGenerativeAI"]).GoogleGenerativeAI,
                "args": {"model":"gemini-2.5-flash", "temperature":0.1, "api_key_env":"GOOGLE_API_KEY"}
            }
        if use_deepseek:
            MODEL_CONFIG["deepseek"] = {
                "class": __import__("langchain_openai", fromlist=["ChatOpenAI"]).ChatOpenAI,
                "args": {"model":"deepseek/deepseek-chat", "temperature":0.1, "base_url":"https://openrouter.ai/api/v1", "api_key_env":"OPENROUTER_API_KEY"}
            }
        if use_cohere:
            MODEL_CONFIG["cohere"] = {
                "class": __import__("langchain_cohere", fromlist=["ChatCohere"]).ChatCohere,
                "args": {"model":"command-r-plus", "temperature":0.1, "api_key_env":"COHERE_API_KEY"}
            }
        chains, reconciler = initialize_models_and_chains(MODEL_CONFIG, st.session_state.retriever, None, None)
        # initialize_models_and_chains expects prompt templates; in this demo, we call chains.invoke(question) directly
        st.session_state.chains = chains
        st.session_state.reconciler = reconciler
        st.success("✅ Models & chains initialized.")

# Run full analysis (calls get_verified_response per task)
from utils import market_snapshot_md

if run_analysis_btn:
    if not st.session_state.retriever:
        st.warning("Upload a PDF first.")
    else:
        st.subheader("📈 Company Snapshot")
        st.markdown(market_snapshot_md(ticker))
        st.subheader("🔎 Automated Analysis")
        for title, instructions in {
            "Growth Analysis": "Find current and previous year 'Revenue' and 'EBITDA'/'EBIT' and compute YoY growth.",
            "Profitability Analysis": "Calculate margins and DuPont ROE.",
            "Liquidity and Solvency": "Current & Quick ratio, interest cover, net debt/EBITDA.",
            "Efficiency and Working Capital": "CCC, DIO, DSO, DPO.",
            "Cash Flow and Dividends": "FCF and payout ratio."
        }.items():
            with st.expander(title):
                prompt = instructions
                final, per_model = get_verified_response(prompt, st.session_state.retriever, st.session_state.chains, st.session_state.reconciler)
                st.markdown(final)
                if st.checkbox(f"Show per-model answers for {title}", key=f"show_models_{title}"):
                    for m, ma in per_model.items():
                        st.markdown(f"**{m}** (confidence: {ma.confidence})\n\n{ma.answer}")

# Q&A
if ask_btn:
    if not st.session_state.retriever:
        st.warning("Upload a PDF first.")
    elif not question_input.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Querying models..."):
            final, per_model = get_verified_response(question_input, st.session_state.retriever, st.session_state.chains, st.session_state.reconciler)
        st.subheader("✅ Verified Answer")
        st.markdown(final)
        if st.checkbox("Show per-model answers", key="show_per_model_q"):
            for m, ma in per_model.items():
                st.markdown(f"**{m}** (confidence: {ma.confidence})\n\n{ma.answer}")

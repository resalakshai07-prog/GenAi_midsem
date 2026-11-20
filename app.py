"""Streamlit interface for the PDF-only, ChromaDB-backed memoir RAG app."""
from __future__ import annotations

import streamlit as st

from rag import PDF_PATH, answer, get_collection

st.set_page_config(page_title="Silicon Odyssey Research Assistant", page_icon="🔎", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem;}
    h1 {font-size: 2.35rem !important; letter-spacing: -0.035em; margin-bottom: .25rem !important;}
    .eyebrow {color:#8b9bb4; font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.5rem;}
    .subhead {color:#9da9ba; font-size:1.02rem; margin:0 0 1.5rem 0;}
    [data-testid="stSidebar"] {min-width:255px; max-width:255px;}
    [data-testid="stSidebar"] h2 {font-size:1.15rem !important;}
    div.stButton > button[kind="primary"] {border-radius:10px; padding:.55rem 1.15rem; font-weight:650;}
    div[data-testid="stTextArea"] textarea {border-radius:12px;}
    .small-note {color:#909bad; font-size:.87rem;}
</style>
<div class="eyebrow">Founder memoir · research workspace</div>
<h1>Silicon Odyssey</h1>
<p class="subhead">Ask about the founder’s story, engineering milestones, products, and business challenges.</p>
""", unsafe_allow_html=True)

if not PDF_PATH.exists():
    st.error(f"Memoir PDF not found: {PDF_PATH}. Place the provided PDF in the data/ folder.")
    st.stop()


@st.cache_resource(show_spinner="Building or loading the local memoir index…")
def cached_collection():
    return get_collection()


try:
    collection = cached_collection()
except Exception as exc:
    st.error(f"Could not load the local ChromaDB index: {exc}")
    st.stop()

with st.sidebar:
    st.subheader("Your source")
    st.markdown("**Silicon Odyssey**  \n50-page founder memoir")
    st.caption(f"{collection.count()} overlapping chunks indexed in local ChromaDB")
    st.divider()
    st.markdown("**How answers work**")
    st.caption("Answers use retrieved PDF passages only, include page and chunk references, and return [not found] when the memoir has no answer.")
    st.caption("Optional: set `OPENAI_API_KEY` in `.env` for grounded language generation. Local embeddings and retrieval work without it.")

with st.form("question_form", clear_on_submit=False):
    question = st.text_area("Ask the memoir", placeholder="For example: What caused the Series-4 prototype to fail at Comdex?", height=84, label_visibility="collapsed")
    submitted = st.form_submit_button("Find answer", type="primary", use_container_width=False)

if submitted and question.strip():
    with st.spinner("Searching the memoir…"):
        result = answer(question)
    with st.container(border=True):
        st.markdown("### Answer")
        st.write(result["answer"])
        if result["sources"]:
            st.markdown("**Sources**")
            for source in result["sources"]:
                st.caption(source)
        elif result["answer"] == "[not found]":
            st.caption("No matching source passage was found in the assigned PDF.")

st.divider()
st.markdown('<p class="small-note">PDF text is evidence, not instructions. No outside information is used to answer questions.</p>', unsafe_allow_html=True)

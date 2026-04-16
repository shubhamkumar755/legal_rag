import sys
import os
import streamlit as st
import tempfile
from transformers import logging
logging.set_verbosity_error()

# ── ENV FIXES ────────────────────────────────────────────────────────────────
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── IMPORT YOUR PIPELINES ────────────────────────────────────────────────────
from scripts.pipeline import run
from batch.document_parser import process_document

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="LegalRAG", page_icon="⚖️", layout="centered")

st.title("LegalRAG")
st.caption("Plain-English explanations of Indian law")
st.divider()

# ── SESSION STATE ────────────────────────────────────────────────────────────
if "doc_result" not in st.session_state:
    st.session_state.doc_result = None

# ── INPUT UI ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["✍️ Type Query", "📄 Upload Document"])

query = ""

# ── TAB 1: DIRECT QUERY ──────────────────────────────────────────────────────
with tab1:
    query = st.text_area(
        "input the clause",
        placeholder="The Parties agree that all confidential information disclosed during the term of this Agreement shall be kept strictly confidential and shall not be disclosed to any third party without prior written consent. This obligation shall survive termination of this Agreement."
    )

# ── TAB 2: DOCUMENT UPLOAD ───────────────────────────────────────────────────
with tab2:
    uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])

    if uploaded_file:

        with st.spinner("Processing document..."):

            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".docx"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name

            # RUN FULL DOCUMENT PIPELINE
            result = process_document(temp_path)

            # Store in session
            st.session_state.doc_result = result

        st.success("✅ Document processed successfully")

# ── EXPLAIN BUTTON ───────────────────────────────────────────────────────────
if st.button("Explain"):

    # ── CASE 1: DOCUMENT MODE ────────────────────────────────────────────────
    if st.session_state.doc_result is not None:

        result = st.session_state.doc_result

        # SUMMARY
        st.subheader("📊 Document Summary")
        st.write(f"**Total Clauses:** {result['total_clauses']}")
        st.write(f"**Processed:** {result['processed']}")
        st.write(f"**Failed:** {result['failed']}")
        st.write(f"**Overall Risk:** {result['overall_risk']}")
        st.write(f"**Average Confidence:** {result['avg_confidence']}")

        st.divider()

        # CLAUSE RESULTS
        st.subheader("📌 Clause Analysis")

        for clause in result["clause_analyses"][:5]:

            if "error" in clause:
                continue

            st.markdown(f"### Clause {clause['clause_id']}")
            st.write(clause["explanation"])

            # CITATIONS
            if clause.get("citations"):
                for c in clause["citations"]:
                    icon = {
                        "verified": "✅",
                        "partial": "⚠️",
                        "not_found": "🚨"
                    }.get(c["status"], "")
                    st.write(f"{icon} {c['act']} — Section {c['section']}")

            # CONFIDENCE
            if isinstance(clause.get("confidence"), dict):
                st.write(clause["confidence"].get("label", ""))

            # RISK
            if clause.get("risk_flag"):
                st.warning(clause["risk_flag"])

            st.divider()

    # ── CASE 2: SINGLE QUERY MODE ────────────────────────────────────────────
    elif query.strip():

        with st.spinner("Searching relevant laws..."):
            result = run(query)

        st.subheader("Explanation")
        st.write(result["explanation"])

        # CITATIONS
        if result.get("citations"):
            st.subheader("Relevant Sections")
            for c in result["citations"]:
                icon = {
                    "verified": "✅",
                    "partial": "⚠️",
                    "not_found": "🚨"
                }.get(c["status"], "")
                st.write(f"{icon} {c['act']} — Section {c['section']}")

        # CONFIDENCE
        st.subheader("Confidence")
        st.write(result["confidence"]["label"])

        # RISK
        if result.get("risk_flag"):
            st.warning(result["risk_flag"])

    else:
        st.warning("Please enter a query or upload a document.")
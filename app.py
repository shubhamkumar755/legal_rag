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

# ── IMPORT YOUR PIPELINE ─────────────────────────────────────────────────────
from pipeline import answer_query, analyze_document

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
        "Enter your legal question or clause",
        placeholder=(
            "The Parties agree that all confidential information disclosed during "
            "the term of this Agreement shall be kept strictly confidential and shall "
            "not be disclosed to any third party without prior written consent. "
            "This obligation shall survive termination of this Agreement."
        ),
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
            st.session_state.doc_result = analyze_document(temp_path)

        st.success("✅ Document processed successfully")

# ── EXPLAIN BUTTON ───────────────────────────────────────────────────────────
if st.button("Explain"):

    # ── CASE 1: DOCUMENT MODE ────────────────────────────────────────────────
    if st.session_state.doc_result is not None:

        clauses = st.session_state.doc_result

        # SUMMARY
        total    = len(clauses)
        failed   = sum(1 for c in clauses if "error" in c)
        processed = total - failed

        # Aggregate confidence labels across successful clauses
        confidence_labels = [
            c["verification_data"].get("confidence", {}).get("label", "")
            for c in clauses
            if "error" not in c and isinstance(c.get("verification_data"), dict)
        ]

        # Aggregate risk flags
        risk_flags = [
            c["verification_data"].get("risk_flag", {})
            for c in clauses
            if "error" not in c
            and isinstance(c.get("verification_data"), dict)
            and c["verification_data"].get("risk_flag")
        ]
        overall_risk = risk_flags[0].get("label", "None") if risk_flags else "None"
        avg_confidence = confidence_labels[0] if confidence_labels else "N/A"

        st.subheader("📊 Document Summary")
        st.write(f"**Total Clauses:** {total}")
        st.write(f"**Processed:** {processed}")
        st.write(f"**Failed:** {failed}")
        st.write(f"**Overall Risk:** {overall_risk}")
        st.write(f"**Average Confidence:** {avg_confidence}")

        st.divider()

        # CLAUSE RESULTS
        st.subheader("📌 Clause Analysis")

        for clause in clauses[:5]:

            if "error" in clause:
                st.markdown(f"### Clause {clause['clause_id']}")
                st.error(f"Failed to analyse: {clause['error']}")
                st.divider()
                continue

            st.markdown(f"### Clause {clause['clause_id']}")
            st.write(clause["explanation"])

            # CITATIONS — stored under verification_data
            vd = clause.get("verification_data", {})
            citations = vd.get("citations", [])
            if citations:
                for c in citations:
                    icon = {"verified": "✅", "partial": "⚠️", "not_found": "🚨"}.get(
                        c.get("status", ""), ""
                    )
                    st.write(f"{icon} {c.get('act', '')} — Section {c.get('section', '')}")

            # CONFIDENCE
            confidence = vd.get("confidence", {})
            if confidence:
                st.write(confidence.get("label", ""))

            # RISK
            risk_flag = vd.get("risk_flag")
            if risk_flag:
                label = risk_flag.get("label") if isinstance(risk_flag, dict) else str(risk_flag)
                st.warning(label)

            # WARNINGS (hallucination / citation drift)
            if clause.get("warnings"):
                for w in clause["warnings"]:
                    st.caption(w)

            st.divider()

    # ── CASE 2: SINGLE QUERY MODE ────────────────────────────────────────────
    elif query.strip():

        with st.spinner("Searching relevant laws..."):
            result = answer_query(query.strip())

        # result is a GeneratorResult dataclass
        vd = result.verification_data or {}

        st.subheader("Explanation")
        st.write(result.response)

        # CITATIONS
        citations = vd.get("citations", [])
        if citations:
            st.subheader("Relevant Sections")
            for c in citations:
                icon = {"verified": "✅", "partial": "⚠️", "not_found": "🚨"}.get(
                    c.get("status", ""), ""
                )
                st.write(f"{icon} {c.get('act', '')} — Section {c.get('section', '')}")

        # CONFIDENCE
        confidence = vd.get("confidence", {})
        if confidence:
            st.subheader("Confidence")
            st.write(confidence.get("label", ""))

        # RISK
        risk_flag = vd.get("risk_flag")
        if risk_flag:
            label = risk_flag.get("label") if isinstance(risk_flag, dict) else str(risk_flag)
            st.warning(label)

        # WARNINGS
        if result.warnings:
            for w in result.warnings:
                st.caption(w)

    else:
        st.warning("Please enter a query or upload a document.")
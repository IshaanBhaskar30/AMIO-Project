import streamlit as st
from market_agents import build_graph
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="AMIO", layout="wide")
st.title("🚀 Autonomous Market Intelligence Orchestrator")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔐 API Keys")

groq_key = st.sidebar.text_input(
    "Groq API Key",
    value=os.getenv("GROQ_API_KEY", ""),
    type="password"
)

tavily_key = st.sidebar.text_input(
    "Tavily API Key",
    value=os.getenv("TAVILY_API_KEY", ""),
    type="password"
)

model_choice = st.sidebar.selectbox(
    "Model",
    ["llama-3.3-70b-versatile",
     "meta-llama/llama-4-scout-17b-16e-instruct"]
)

# ============================================================
# QUERY INPUT
# ============================================================

query = st.text_area("Enter Company or Industry Query")


# ============================================================
# MAIN EXECUTION
# ============================================================

if st.button("Generate Report"):

    if not groq_key or not tavily_key:
        st.error("Please enter both API keys.")
    elif not query.strip():
        st.error("Please enter a query.")
    else:
        with st.spinner("Running Autonomous Intelligence System..."):

            graph = build_graph(groq_key, tavily_key, model_choice)

            result = graph.invoke({
                "query": query,
                "sources": [],
                "refinement_count": 0
            })

        report = result["final_report"]
        confidence = result["confidence"]

        # ========================================================
        # DISPLAY REPORT
        # ========================================================

        st.subheader("📊 Executive Report")
        st.markdown(report["content"])

        st.subheader("🧠 Confidence Assessment")
        st.json(confidence)

        # ========================================================
        # PDF EXPORT
        # ========================================================

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Market Intelligence Report", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * inch))

        # Full Report Content
        elements.append(Paragraph("Executive Report", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(report["content"].replace("\n", "<br/>"), styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        # Confidence Section
        elements.append(Paragraph("Confidence Assessment", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"Score: {confidence['score']}", styles["Normal"]))
        elements.append(Paragraph(confidence["reasoning"], styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        # Sources
        elements.append(Paragraph("Sources", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        for s in report["sources"]:
            elements.append(Paragraph(s, styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))

        doc.build(elements)

        st.download_button(
            label="📥 Download Report as PDF",
            data=buffer.getvalue(),
            file_name="market_intelligence_report.pdf",
            mime="application/pdf"
        )

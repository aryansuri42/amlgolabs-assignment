import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
from rag_pipeline import RagPipeline

# Initialize the RAG pipeline
@st.cache_resource
def load_rag_pipeline():
    return RagPipeline()


rag_pipeline = load_rag_pipeline()

# App layout
st.set_page_config(page_title="RAG App", layout="wide")
st.title("📚 Retrieval-Augmented Generation (RAG) App")
st.markdown("Ask a question based on the **AI Training Document**.")

# Show the model name
st.sidebar.header("Model Info")
st.sidebar.write(f"**LLM Model:** {rag_pipeline.model_name}")

# Input box for user query
query = st.text_input("Enter your query:", placeholder="e.g., What is eBay's policy on returns?")

if query:
    with st.spinner("Generating answer..."):
        context, answer = rag_pipeline.Pipeline(query)

    if "[/INST]" in answer:
        clean_answer = answer.split("[/INST]")[-1].strip()
    else:
        clean_answer = answer.strip()

    # Output section
    st.subheader("📌 Answer:")
    st.write(clean_answer)

    with st.expander("📄 Context used from the document"):
        st.write(context)

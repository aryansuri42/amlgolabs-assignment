Retrieval-Augmented Generation (RAG) Chatbot
This project implements a Streamlit-based Retrieval-Augmented Generation (RAG) system that allows users to query information grounded in a specific document (e.g., eBay User Agreement). It combines FAISS vector search, SentenceTransformers embeddings, and a quantized Mistral-7B model via Hugging Face for fast and accurate answers.

Project Architecture and Flow
mathematica
Copy
Edit
PDF Document
    │
    └──► Preprocessing
             └──► Text Chunking (RecursiveCharacterTextSplitter)
                     └──► Embedding Generation (SentenceTransformer)
                             └──► Vector DB Creation (FAISS)
                                     └──► Query Input
                                             ├──► Top-k Similar Chunks Retrieved
                                             └──► Passed as Context to Mistral-7B LLM (Quantized)
                                                     └──► Final Answer Generated with Context
Document Loader: PyPDFLoader (via Langchain)

Chunking Strategy: RecursiveCharacterTextSplitter (chunk size 300, overlap 50)

Embedding Model: all-mpnet-base-v2 (SentenceTransformers)

Vector Store: FAISS for efficient similarity search

LLM: Quantized mistralai/Mistral-7B-Instruct-v0.1 (4-bit)

Frontend: Streamlit UI for interactive queries

Setup Instructions
Clone the repository and install dependencies:

bash
Copy
Edit
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
Ensure that you have access to the quantized Mistral model via Hugging Face, or adjust to use a different model if needed.

Steps to Run
1. Preprocessing
bash
Copy
Edit
python src/document_preprocessing.py
Loads the input PDF using Langchain's PyPDFLoader.

Chunks the text into manageable segments for embedding.

2. Create Embeddings and Build Vector DB
bash
Copy
Edit
python src/vector_searching.py
Uses the SentenceTransformer model to embed the text chunks.

Stores the embeddings into a FAISS vector database.

3. Load the RAG Pipeline
bash
Copy
Edit
python src/rag_pipeline.py
Builds the retrieval pipeline using FAISS and Hugging Face Transformers.

Defines the prompt template and LLMChain.

Running the Chatbot with Streaming
bash
Copy
Edit
streamlit run app.py
Enter your question in the input box.

The system will retrieve the top relevant document chunks and generate an answer using the LLM.

The context retrieved will also be displayed for transparency.

Model and Embedding Choices
Component	Choice	Justification
Embedding Model	all-mpnet-base-v2	Strong semantic similarity, widely used for dense vector search
LLM	mistralai/Mistral-7B-Instruct-v0.1	Instruction-tuned, quantized for low memory footprint
Vector Store	FAISS	Fast and scalable similarity search on dense vectors
Chunking Method	RecursiveCharacterTextSplitter	Ensures semantic coherence and overlap for better context recall

Sample Queries and Outputs
Query Example
vbnet
Copy
Edit
What is eBay's policy on returns?
Output Example
vbnet
Copy
Edit
eBay allows sellers to specify their return policy in listings. However, all sellers must comply with eBay's Money Back Guarantee which ensures buyers can return items not received or not as described, within a specific time frame.
Context Example
kotlin
Copy
Edit
eBay’s Money Back Guarantee ensures buyers can return items not as described. The seller may specif
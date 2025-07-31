from vector_searching import VectorDBSearching
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import transformers
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough

class RagPipeline(VectorDBSearching):

    def __init__(self):

        """
        The RagPipeline class is responsible for building a Retrieval-Augmented Generation (RAG) pipeline.
        It inherits from VectorDBSearching, which provides document preprocessing and vector search capabilities.
        This class loads a quantized LLM (Mistral-7B) using HuggingFace Transformers, embeds the documents using 
        sentence-transformers, and generates responses based on retrieved context using LangChain.
        """

        super().__init__()
        self.db = FAISS.from_documents(self.components[1], 
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
        self.model_name='mistralai/Mistral-7B-Instruct-v0.1'
        self.model_config = AutoConfig.from_pretrained(self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.use_4bit = True
        self.bnb_4bit_compute_dtype = "float16"
        self.bnb_4bit_quant_type = "nf4"
        self.use_nested_quant = False
        self.bnb_config = BitsAndBytesConfig(
                        load_in_4bit=self.use_4bit,
                        bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                        bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                        bnb_4bit_use_double_quant=self.use_nested_quant,
                    )
        self.model_llm = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            quantization_config=self.bnb_config,
                        )
        
    def Pipeline(self, query):
        """
        Executes the RAG pipeline by retrieving context from the FAISS DB and generating 
        an answer using the quantized Mistral model.
        
        Args:
            query (str): User query
        
        Returns:
            list: [retrieved context, generated response]
        """

        self.text_generation_pipeline = transformers.pipeline(
                model=self.model_llm,
                tokenizer=self.tokenizer,
                task="text-generation",
                temperature=0.2,
                repetition_penalty=1.1,
                return_full_text=True,
                max_new_tokens=300,
            )
        self.prompt_template = """
                [INST]
                You are a helpful assistant. Use the information from the context below to answer the user's question.
                If the answer is not in the context, say "The document does not contain that information."

                Context:
                {context}

                Question: {question}
                [/INST]
                """
        
        self.mistral_llm = HuggingFacePipeline(pipeline=self.text_generation_pipeline)

        # Create prompt from prompt template 
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.prompt_template,
        )

        # Create llm chain 
        self.llm_chain = LLMChain(llm=self.mistral_llm, prompt=self.prompt)
        self.retriever = self.db.as_retriever()
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        self.rag_chain = ( 
        {"context": self.retriever|format_docs, "question": RunnablePassthrough()}
            | self.llm_chain
        )
        answer = self.rag_chain.invoke(query)
        return [answer['context'], answer['text']]
    
__all__ = ["RagPipeline"]

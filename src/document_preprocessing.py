from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentPreprocessing:

    def __init__(self):
        self.file_path = "C:\\Users\\ARYAN SURI\\Desktop\\Amlgo Labs Assignment\\data\\AI Training Document.pdf"
        self.loader = PyPDFLoader(self.file_path)
        self.docs = self.loader.load()

    def RecursiveSplitter(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        self.documents  = self.text_splitter.split_documents(self.docs)
        self.texts = [doc.page_content for doc in self.documents]
        return [self.texts, self.documents]
    
__all__ = ["DocumentPreprocessing"]
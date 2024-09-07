import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated, Any
from fastapi.responses import StreamingResponse

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceSplitter
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
GROQ_API_KEY = os.getenv("api_key")
MODEL_NAME = os.getenv("model_name")

class PrepareRagRequest(BaseModel):
    folder_path: Annotated[str, Field(..., description="The path to the file to be processed")]
    
class PrepareRagResponse(BaseModel):
    message: str
    query_engine: Any
    
class AskRequest(BaseModel):
    query: Annotated[str, Field(..., description="The query to be processed")]
    
class AskResponse(BaseModel):
    message: str
    

class Chatbot:
    def __init__(self):
        self.query_engine = None
        self.folder_path = None
        
    def prepare_index(self, folder_path: str) -> PrepareRagResponse:
        self.folder_path = folder_path
        if os.path.exists("./db"):
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            llm = Groq(model=MODEL_NAME, api_key=GROQ_API_KEY)
            # Load LLM and embedding models
            Settings.llm = llm
            Settings.embed_model = embed_model
            
            storage_context = StorageContext.from_defaults(persist_dir="./db")
            index = load_index_from_storage(storage_context=storage_context)
            

            # Create a query engine from the existing index
            query_engine = index.as_query_engine(llm=llm, streaming=True, similarity_top_k=4)
            return query_engine
    
        else:
            # Load Files
            reader = SimpleDirectoryReader(input_dir=self.folder_path)
            documents = reader.load_data()
            print(f"Total Documents: {len(documents)}")

            # Chucking Data
            text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
            nodes = text_splitter.get_nodes_from_documents(documents=documents, show_progress=True)
            print(f"Total Nodes: {len(nodes)}")

            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

            llm = Groq(model=MODEL_NAME, api_key=GROQ_API_KEY)
            Settings.llm = llm
            Settings.embed_model = embed_model


            vector_index = VectorStoreIndex.from_documents(documents=documents, show_progress=True, node_parser=nodes)

            vector_index.storage_context.persist(persist_dir="./db")

            storage_context = StorageContext.from_defaults(persist_dir="./db")

            index = load_index_from_storage(storage_context)

            query_engine = index.as_query_engine(llm=llm, streaming=True, similarity_top_k=4)

            return query_engine
    
    def ask(self, query: str) -> AskResponse:
        if self.query_engine is None:
            return AskResponse(message="Query engine not initialized yet")
        
        result = self.query_engine.query(query)

        # If result is a StreamingResponse, handle streaming response aggregation
        if isinstance(result, StreamingResponse):
            # Aggregate response (collect all parts of the streaming response)
            full_response = ""
            for part in result.iter_content():
                full_response += part.decode()  # assuming the content is in bytes and needs decoding
            return AskResponse(message=full_response)
        else:
            # If the result is already a string or expected format
            return AskResponse(message=str(result))
    
    def prepare_rag(self, folder_path: str) -> PrepareRagResponse:
        self.query_engine = self.prepare_index(folder_path)
        return PrepareRagResponse(message="RAG prepared successfully", query_engine=self.query_engine)
    
    def ask_rag(self, query: str) -> AskResponse:
        return self.ask(query)

if __name__ == "__main__":
    chatbot = Chatbot()
    print("*" * 100)
    print("Welcome to the Phoenix Chatbot")
    print("*" * 100)
    folder_path = input("Enter the Folder Path: ")
    query_engine = chatbot.prepare_rag(folder_path)
    print("*" * 100)
    print("You can ask questions about the document")
    print("Type 'bye' to exit")
    print("*" * 100)
    while (query := input("Enter your query: ")) != "bye":
        print("*" * 100)
        result = chatbot.ask_rag(query)
        print(result)
        print("*" * 100)


from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file("Data/")
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "bns-chatbot"

# Check if index exists
existing_indexes = pc.list_indexes()
if index_name not in [index.name for index in existing_indexes]:
    # Create index if it doesn't exist
    pc.create_index(
        name=index_name,
        dimension=384,  # Adjust based on your embedding model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )


# Embedding each chunk and adding the embeddings into Pinecone index
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)
from fastapi import FastAPI
from pydantic import BaseModel
import re
from pymongo import MongoClient
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever
from fastapi.middleware.cors import CORSMiddleware
from keybert import KeyBERT  # For keyword extraction
from fastapi import HTTPException,Request
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
    # "http://example.com",  # Add specific domains as needed
    # "*"  # Uncomment to allow all origins (not recommended in production)
]

# Adding CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,  # Allows cookies to be included in cross-origin requests
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allows all headers (custom headers, etc.)
)

# MongoDB Connection
client = MongoClient("mongodb+srv://aryanwork10:7AEMpdRmjnUJl01Z@cluster0.0hhol.mongodb.net/BTPRecommendationDB?retryWrites=true&w=majority&appName=Cluster0")  # Replace with your actual MongoDB URI
db = client["BTPRecommendationDB"]  # Replace with your database name
collection = db["professors"]  # Replace with your collection name


keyword_extractor = KeyBERT(model='mixedbread-ai/mxbai-embed-large-v1')  # You can adjust the model

# Define request model for abstract submission
class AbstractRequest(BaseModel):
    professor_id: str  # Identifier for the professor
    abstract: str      # Abstract of the paper


# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
    else:
        text = ""
    return text

# Fetch and preprocess faculty data from MongoDB
def fetch_data_from_mongo():
    data = list(collection.find({}, {"_id": 1, "name": 1, "researchAreas": 1, "researchTechnologies": 1, "email": 1, "department": 1, "profilePic": 1}))  # Include only necessary fields
    for item in data:
        item['_id'] = str(item['_id'])  # Convert ObjectId to string
        item['researchAreas'] = preprocess_text(item.get('researchAreas', ''))
        item['researchTechnologies'] = preprocess_text(item.get('researchTechnologies', ''))
        item['email'] = str(item['email'])  # Convert email to string
        item['department'] = str(item['department'])  # Convert department to string
        item['profilePic'] = str(item['profilePic'])
    return data


# Initialize HuggingFace embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

data=fetch_data_from_mongo()
# Filter out empty or invalid documents before creating the BM25 retriever
documents = [
    f"{item['researchAreas']} {item['researchTechnologies']}"
    for item in data if item['researchAreas'] or item['researchTechnologies']
]

# metadata = [{"Name": item["name"]} for item in data]
# Update metadata to include firstName and lastName
# metadata = [{"firstName": item["name"]["firstName"], "lastName": item["name"]["lastName"]} for item in data]
metadata = [item for item in data]

# Check if documents is empty
if not documents:
    raise ValueError("No valid documents found for BM25 retriever.")

# Initialize BM25 Retriever if documents are non-empty
from langchain.retrievers.bm25 import BM25Retriever
if documents:
    bm25_retriever = BM25Retriever.from_texts(documents, metadatas=metadata)
else:
    bm25_retriever = None  # or handle accordingly if BM25 is essential

# Build FAISS index with metadata
vector_store = FAISS.from_texts(documents, embeddings, metadatas=metadata)

# Initialize the ensemble retriever, excluding BM25 if it was not created
ensemble_retrievers = [vector_store.as_retriever()]
if bm25_retriever:
    ensemble_retrievers.append(bm25_retriever)

# Adjust weights if BM25 is included or not
weights = [1] if not bm25_retriever else [0.5, 0.5]

ensemble_retriever = EnsembleRetriever(
    retrievers=ensemble_retrievers,
    weights=weights
)



# # Prepare data for FAISS and BM25 retrievers
# data = fetch_data_from_mongo()
# documents = [
#     f"{item['researchAreas']} {item['keywords']}"
#     for item in data
# ]

# # metadata = [{"Name": item["name"]} for item in data]
# # Update metadata to include firstName and lastName
# metadata = [{"firstName": item["name"]["firstName"], "lastName": item["name"]["lastName"]} for item in data]

# # Build FAISS index with metadata
# vector_store = FAISS.from_texts(documents, embeddings, metadatas=metadata)

# # Initialize BM25 Retriever
# from langchain.retrievers.bm25 import BM25Retriever
# bm25_retriever = BM25Retriever.from_texts(documents, metadatas=metadata)

# # Initialize the ensemble retriever
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[vector_store.as_retriever(), bm25_retriever],
#     weights=[0.5, 0.5]
# )

# Combined search function using EnsembleRetriever
def combined_search(project_description, top_n=3):
    results = ensemble_retriever.get_relevant_documents(project_description)
    top_faculties = [
        {
            **result.metadata,
            "_id": str(result.metadata["_id"]),  # Ensure _id is serialized
        }
        for result in results[:top_n]
    ]
    return top_faculties


# Define request model
class SearchRequest(BaseModel):
    project_description: str
    top_n: int = 3

# FastAPI route for combined search
@app.post("/search")
def search_faculties(request: SearchRequest):
    preprocessed_description = preprocess_text(request.project_description)
    result = combined_search(preprocessed_description, request.top_n)
    return {"top_faculties": result}

# Function to extract keywords from the abstract
def extract_keywords(abstract, top_n=10):
    keywords = keyword_extractor.extract_keywords(
        abstract,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n
    )
    return [kw[0] for kw in keywords]  # Only return the keyword strings


@app.post("/add_keywords")
def add_keywords_to_professor(request: AbstractRequest):
    abstract = preprocess_text(request.abstract)
    if not abstract:
        raise HTTPException(status_code=400, detail="Abstract cannot be empty.")

    keywords = extract_keywords(abstract)
    if not keywords:
        raise HTTPException(status_code=400, detail="No keywords could be extracted.")

    # Find the professor document
    professor = collection.find_one({"email": request.professor_id})

    if professor:
        current_research = professor.get("researchTechnologies", "")

        # Ensure it's a string before updating
        if isinstance(current_research, str):
            current_keywords = current_research.split(", ") if current_research else []
            updated_keywords = current_keywords + keywords
            updated_research_technologies = ", ".join(updated_keywords)
        else:
            updated_research_technologies = ", ".join(current_research + keywords)

        # Update the document in MongoDB
        result = collection.update_one(
            {"email": request.professor_id},
            {"$set": {"researchTechnologies": updated_research_technologies}}
        )

        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Professor not found or no changes made.")

        return {"message": "Keywords successfully added.", "keywords": keywords}

    else:
        raise HTTPException(status_code=404, detail="Professor not found.")

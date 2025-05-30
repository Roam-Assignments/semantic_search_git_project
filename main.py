from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import subprocess
import os

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

model = SentenceTransformer("all-MiniLM-L6-v2")
data = []
index = None
DB_PATH = "semantic-db/player_abilities.json"

def load_data():
    global data, index
    with open(DB_PATH) as f:
        data = json.load(f)
    descriptions = [d["Description"] for d in data]
    vectors = model.encode(descriptions)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))

@app.on_event("startup")
def startup_event():
    load_data()

@app.post("/search")
def search(request: QueryRequest):
    query_vec = model.encode([request.query])
    D, I = index.search(np.array(query_vec), request.top_k)
    results = [data[i] for i in I[0]]
    return {"results": results}

@app.post("/reload")
def reload():
    load_data()
    return {"status": "reloaded"}

@app.post("/sync")
def sync_and_reload():
    try:
        subprocess.run(["git", "-C", "semantic-db", "pull"], check=True)
        load_data()
        return {"status": "updated and reloaded from Git"}
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}

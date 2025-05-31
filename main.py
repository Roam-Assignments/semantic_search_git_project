
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import subprocess

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

model = SentenceTransformer("all-MiniLM-L6-v2")

# Global storage
data_store = {}
index_store = {}
files = {
    "abilities": "semantic-db/player_abilities.json",
    "shaders": "semantic-db/shaders.json",
    "behaviours": "semantic-db/asset_behaviours.json",
    "objectives": "semantic-db/objectives.json"
}

def load_data():
    for key, path in files.items():
        with open(path) as f:
            records = json.load(f)
            valid_records = [r for r in records if "Description" in r]
            descriptions = [r["Description"] for r in valid_records]
            if len(valid_records) < len(records):
                print(f"[WARN] {key}: {len(records) - len(valid_records)} entries skipped (missing 'Description')")
            vectors = model.encode(descriptions)
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(np.array(vectors))
            data_store[key] = valid_records
            index_store[key] = index

@app.on_event("startup")
def startup_event():
    load_data()

@app.post("/search/abilities")
def search_abilities(req: QueryRequest):
    return run_search("abilities", req)

@app.post("/search/shaders")
def search_shaders(req: QueryRequest):
    return run_search("shaders", req)

@app.post("/search/behaviours")
def search_behaviours(req: QueryRequest):
    return run_search("behaviours", req)

@app.post("/search/objectives")
def search_objectives(req: QueryRequest):
    return run_search("objectives", req)

@app.get("/abilities")
def get_all_abilities():
    return {"abilities": data_store.get("abilities", [])}

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

def run_search(key: str, req: QueryRequest):
    query_vec = model.encode([req.query])
    D, I = index_store[key].search(np.array(query_vec), req.top_k)
    results = [data_store[key][i] for i in I[0]]
    return {"results": results}

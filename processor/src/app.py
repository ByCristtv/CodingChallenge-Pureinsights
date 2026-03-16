import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Generator

from elasticsearch import Elasticsearch, helpers
from sentence_splitter import SentenceSplitter
from sentence_transformers import SentenceTransformer

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
INPUT_DIR = os.getenv("INPUT_DIR", "/app/input")
INDEX_NAME = os.getenv("INDEX_NAME", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))

# Simple sentence embeddings model
model = SentenceTransformer(EMBEDDING_MODEL)

# Sentence Splitter
splitter = SentenceSplitter(language="en")

def create_index(es: Elasticsearch, index_name: str) -> None:
    # Create the index if it does not exist.
    if es.indices.exists(index=index_name):
        return

    mapping = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "title": {"type": "text"},
                "description": {"type": "text"},
                "subjects": {"type": "keyword"},             
                # TODO: Complete the mapping with the required fields and types. (x)
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384, # Adjust the dimensions where required.
                    "index": True,
                    "similarity": "cosine"

                }
            }
        }
    }

    es.indices.create(index=index_name, body=mapping)
    print(f"Created index: {index_name}")

def load_json_files(input_dir: str) -> List[Dict[str, Any]]:
    documents = []
    for path in Path(input_dir).rglob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            try:
                documents.append(json.load(f))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {path}: {e}")
            
    return documents

def split_into_chunks(text: str, max_sentences: int = 5) -> List[str]:
    # Split the text into small chunks.
    sentences = splitter.split(text)
    chunks = []

    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def generate_embedding(text: str) -> List[float]:
    # TODO: Create the required code to generate text embeddings. (X)
    embedding = model.encode(text)
    return embedding.tolist()

def get_bulk_actions(index_name: str, processed_chunks: List[Dict[str, Any]]) -> Generator:
    for chunk in processed_chunks:
        yield {
            "_index": index_name,
            "_source": chunk
        }
def process_and_index_all(es: Elasticsearch, index_name: str, documents: List[Dict[str, Any]]) -> None:
    all_processed_chunks = []
    text_to_embed = []
    print("Processing documents")
    for doc in documents:
        doc_id = doc.get("id")
        description = doc.get("description", "")
        if not doc_id or not description:
            continue
        clean_description = re.sub(r'[^\x00-\x7F]+', ' ', description)
        clean_subjects = [str(s).upper() for s in doc.get("subjects", [])]
        chunks = split_into_chunks(clean_description)
        for idx, chunk in enumerate(chunks):
            all_processed_chunks.append({
                "doc_id": str(doc_id),
                "chunk_id": f"{doc_id}-{idx}",
                "title": doc.get("title", ""),
                "description": chunk,
                "subjects": clean_subjects
            })
            text_to_embed.append(chunk)
    if not text_to_embed:
        print("No valid documents to process.")
        return
    print("Generating embeddings")
    embeddings = model.encode(text_to_embed, batch_size=BATCH_SIZE)
    final_docs = []
    for meta, embedding in zip(all_processed_chunks, embeddings):
        meta["embedding"] = embedding.tolist()
        final_docs.append(meta)
    for i in range(0, len(final_docs), BATCH_SIZE):
        batch = final_docs[i:i + BATCH_SIZE]
        helpers.bulk(es, get_bulk_actions(index_name, batch))

def semantic_search(es: Elasticsearch, index_name: str, query_text: str, k: int = 3) -> Dict[str, Any]:
    # Query to perform semantic search
    query_vector = model.encode(query_text).tolist()

    body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": 10
        },
        "_source": ["doc_id", "title", "description", "subjects"]
    }

    return es.search(index=index_name, body=body)


def main() -> None:
    es = Elasticsearch(ELASTICSEARCH_URL)
    if not es.ping():
        print(f"Could not connect to Elasticsearch")
        return
    create_index(es, INDEX_NAME)
    print("Loading documents...")
    documents = load_json_files(INPUT_DIR)

    if not documents:
        print("No JSON files found.")
        return

    process_and_index_all(es, INDEX_NAME, documents)

    print("Semantic search: examples")

    # TODO: Create several semantic search queries and print the results. (X)
    # Use the function semantic_search()
    queries = [
        "What are the main topics covered in the document?",
        "Summarize the key points of the document.",
        "What subjects are discussed in the document?",
        "What is the main focus of the document?",
        "What are the key findings or conclusions of the document?"
    ]

    for q in queries:
        print(f"Query: {q}")
        results = semantic_search(es, INDEX_NAME, q)
        for hit in results.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            print(f"Doc ID: {source.get('doc_id')}, Title: {source.get('title')}, Description: {source.get('description')[:100]}..., Subjects: {source.get('subjects')}")
        print("\n")


if __name__ == "__main__":    
    main()
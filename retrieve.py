import os
import json
import logging

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

load_dotenv()

CACHE_FILE = "cache/vector_store_cache.json"
model = "gpt-5.4-nano-2026-03-17"
client = OpenAI()

_vector_store = None


def create_vector_store():
    global _vector_store

    if _vector_store is not None:
        return _vector_store

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            vector_store_cache = json.load(f)

            try:
                _vector_store = client.vector_stores.retrieve(vector_store_cache["id"])
            except Exception as e:
                logging.warning(f"Problem retrieving the vector store. {e}")

            return _vector_store

    logging.info("Creating a new vector store...")
    with open("sample_data/houses.json", "rb") as f:
        house_file = client.files.create(file=f, purpose="assistants")

    _vector_store = client.vector_stores.create(name="houses_knowledge_base")

    client.vector_stores.files.create(
        vector_store_id=_vector_store.id, file_id=house_file.id
    )

    with open(CACHE_FILE, "w") as f:
        json.dump({"id": _vector_store.id}, f)

    return _vector_store


def get_vector_store_id():
    return create_vector_store().id

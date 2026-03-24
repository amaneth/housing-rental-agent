import logging

from dotenv import load_dotenv
from openai import OpenAI

from retrieve import get_vector_store_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    
)

logger = logging.getLogger(__name__)

load_dotenv()

model = "gpt-5.4-nano-2026-03-17"
client = OpenAI()


vector_store_id = get_vector_store_id()

query = "I want to rent an apartment with 1 bedroom under the price of 1000. I have a also a dog pet."


response = client.responses.create(
    model=model,
    input=[
        {"role": "system", "content": "You're a house rental agent"},
        {
            "role": "user",
            "content": query,
        },
    ],
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
        }
    ],
)

# print(json.dumps(response.model_dump(), indent=2))
print(response.output_text)

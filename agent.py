import json
import logging

from dotenv import load_dotenv
from openai import OpenAI

from retrieve import get_vector_store_id
from tools import calculate_distance

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)

load_dotenv()

model = "gpt-5.4"
# model = "gpt-5.4-mini"
client = OpenAI()


vector_store_id = get_vector_store_id()

house_search_tool = {
    "type": "file_search",
    "vector_store_ids": [vector_store_id],
}

distance_tool = {
    "type": "function",
    "name": "calculate_distance",
    "description": "Get the distance between two points",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "lat1": {"type": "number"},
            "lon1": {"type": "number"},
            "lat2": {"type": "number"},
            "lon2": {"type": "number"},
        },
        "additionalProperties": False,
        "required": ["lat1", "lon1", "lat2", "lon2"],
    },
}

SYSTEM_PROMPT = """
You are a house rental agent.

Your goal is to recommend houses that best match the user's requirements based on the available listings.

Guidelines:
- Understand the user's intent (location, budget, bedrooms, type, etc.).
- Prioritize houses that match the requested neighborhood or are close to it.
Distance calculation guidelines:
    - Do NOT calculate distance for every house by default.
    - First identify relevant nearby locations.
    - Then evaluate only the most relevant neighborhood houses within closest locations.
    - Use the calculate_distance tool only when necessary to compare or rank nearby options.

Output:
- Provide clear and concise recommendations.
- Mention neighborhood, key features, and why the house fits the user's needs.
- If helpful, include distance context (e.g., “3 km from CMC”).

Always aim to give practical and relevant recommendations, not just the closest ones.
"""


def call_function(name, args):
    if name == "calculate_distance":
        return calculate_distance(**args)
    raise ValueError(f"Unknow function name: {name}")


def agent(messages):
    model_response = client.responses.create(
        model=model,
        input=messages,
        tools=[house_search_tool, distance_tool],
    )
    messages.extend(model_response.output)
    while True:
        function_calls = [
            item for item in model_response.output if item.type == "function_call"
        ]

        if not function_calls:
            break

        logging.debug("Running the call funcitons loop...")
        for tool_call in function_calls:
            logging.info(f"Calling the function: {tool_call.name}")
            name = tool_call.name
            args = json.loads(tool_call.arguments)
            result = call_function(name, args)

            logging.info(f"Function call result: {result}")

            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(result),
                }
            )

        # print(f"The content message after tool calls: {messages}")
        model_response = client.responses.create(
            model=model,
            input=messages,
            tools=[house_search_tool, distance_tool],
        )
        messages.extend(model_response.output)
    return messages, model_response


def main():
    messages = []

    while True:
        user_message = input("How can I help you with?\n" if not messages else "")

        if user_message.lower() == "exit":
            break

        if not messages:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        else:
            # messages = messages[:15]
            messages += [{"role": "user", "content": user_message}]

        try:
            messages, model_response = agent(messages)
            logger.info(f"Agent: {model_response.output_text}")
        except Exception as e:
            logger.warning(f"Problem getting responses from the AI model: {e}")

        # context += [{"role": "assistant", "content": response.output_text}]
        # print(json.dumps(response.model_dump(), indent=2))
        # print(context)


if __name__ == "__main__":
    main()

import requests # type: ignore
from langprompt import Prompt
from langprompt.llms.openai import OpenAI
from langprompt.cache import SQLiteCache

def get_weather(latitude: float, longitude: float) -> float:
    """Get the current temperature in Celsius"""
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']


tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for provided coordinates in celsius.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"}
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": True
    }
}]


prompt = Prompt[str, str](
    template="""
    <|system|>
    You are a weather expert. You are given a location and you need to provide the current temperature in celsius.
    <|end|>

    <|user|>
    What is the current temperature in {{input}}?
    <|end|>
    """
)

provider = OpenAI(model="gpt-4o-mini", cache=SQLiteCache())

response = provider.chat(prompt.parse("New York"), tools=tools)
print(response)
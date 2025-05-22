from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define a JSON Schema dict for reviews
review_schema = {
    "title": "review",
    "type": "object",
    "properties": {
        "summary":   {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive","neutral","negative"]}
    },
    "required": ["summary","sentiment"],
    "additionalProperties": False,
}

# Bind the schema to the model
structured_model = model.with_structured_output(review_schema)

# Invoke on free-form text
result = structured_model.invoke(
    """The Hardware is great,but the software feels bloated. There are too many pre-installed apps
            that I can't remove, Also the UI looks outdated compared to other brands. Hoping for a software 
            update to fix this."""
)
print(type(result))

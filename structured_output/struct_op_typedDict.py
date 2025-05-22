from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated,Optional
from dotenv import load_dotenv

load_dotenv()

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# TypedDict schema
class review(TypedDict):
    key_themes : Annotated[list[str],""]
    summary:Annotated[str,""]
    sentiment:Annotated[str,""]
    pros: Annotated[Optional[list[str]],""]
    cons: Annotated[Optional[list[str]],""]

# Define a JSON Schema dict for reviews
review_schema = {
    "title": "review",
    "type": "object",
    "properties": {
        "key_themes":{
            "type":"array",
            "description":"Write down all the key themes discussed in the review in a list",
            "items":{"type":"string"}
        },
        "summary":   {
            "type": "string",
            "description":"A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "description":"Return sentiment of the review either negative,positive or neutral",
            "enum": ["positive","neutral","negative"]
        },
        "pros":{
            "type":"string",
            "description": "Write down all the pros inside a list",
            "items":{"type":"string"},
            "nullable" : True     #optional field
        },
        "cons":{
            "type": "array",
            "description":"Write down all the cons inside a list",
            "items":{"type":"string"},
            "nullable":True  #optional field
        }
    },
    "required": ["key_themes","summary","sentiment"],
    "additionalProperties": False,
}

# Bind the schema to the model
structured_model = model.with_structured_output(review_schema)

# Invoke on free-form text
result = structured_model.invoke(
    """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Cons:
Bulky and heavy-not great for one hand use
Bloatware still exists in One UI
Expensive compared to competitors
"""
)
print(result)
print(type(result))

review_data = result[0]['args']

print("Summary: ",review_data['summary'])
print("Sentiment: ",review_data['sentiment'])
print("Key Themes: ",review_data['key_themes'])

"""use get() function for pros and cons, because the are optional 
and hence no guarantee that they will be present while summary, sentiment, key_themes are required
so they are confirmed to be present in the result data (or review data)
"""
print("Pros: ",review_data.get('pros'))
print("Cons: ",review_data.get('cons'))
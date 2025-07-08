from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  
    google_api_key = os.getenv("GOOGLE_API_KEY")
)

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type= "standard_deviation",
    breakpoint_threshold_amount=1
)

sample = """
    She sat in the darkened room waiting. It was now a standoff. 
    He had the power to put her in the room, but not the power to 
    make her repent. It wasn't fair and no matter how long she had 
    to endure the darkness, she wouldn't change her attitude. 
    At three years old, Sandy's stubborn personality had already 
    bloomed into full view.Green vines attached to the trunk of the
    tree had wound themselves toward the top of the canopy. Ants used 
    the vine as their private highway, avoiding all the creases and 
    crags of the bark, to freely move at top speed from top to bottom 
    or bottom to top depending on their current chore. At least this was 
    the way it was supposed to be. Something had damaged the vine overnight 
    halfway up the tree leaving a gap in the once pristine ant highway.
    "Explain to me again why I shouldn't cheat?" he asked. "All the others 
    do and nobody ever gets punished for doing so. I should go about being 
    happy losing to cheaters because I know that I don't? That's what you're 
    telling me?"
"""

chunks = splitter.create_documents([sample])
print(len(chunks))
print(chunks)

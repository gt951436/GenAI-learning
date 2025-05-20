from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Use a hosted, deployed repo so provider_mapping isn't empty:
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")
user_input = st.text_input("Enter your prompt")

if st.button('Summarize'):
    res = model.invoke(user_input)
    st.write(res.content)

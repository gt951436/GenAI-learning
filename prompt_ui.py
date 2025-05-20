from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

# Use a hosted, deployed repo so provider_mapping isn't empty:
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")
paper_input = st.selectbox("Select Research paper name",["Attention is all you need","BERT: Pre-training of Deep Bidierctional Transformers","GPT-3: Language Models are few-shot learners","Diffusion Models beat GANs on Image Synthesis"])
style_input = st.selectbox("Select Explanation Style",["Beginner-Friendly","Technical","Code-Oriented","Mathemtical"])
length_input = st.selectbox("Select Explanation Length",["Short (1-2 paragraphs)","Medium (3-5 paragraphs)","Long (detailed explanation)"])

#load prompt template
template = load_prompt("template.json")

if st.button('Summarize'):
    chain = template|model
    
    res = chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    })
    st.write(res.content)

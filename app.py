import os
from dotenv import load_dotenv
import streamlit as st
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

load_dotenv()
model = ChatOpenAI(model="gpt-4o")
prompt = hub.pull("rlm/rag-prompt")

@st.cache_resource
def get_chain():
    loader = WikipediaLoader(query="歴史を面白く学ぶコテンラジオ", load_max_docs=1, lang="jp")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain

st.title("歴史を面白く学ぶコテンラジオ")
q = st.text_input("質問")
if st.button("質問"):
    chain = get_chain()
    answer = chain.invoke(q)
    st.write(answer)

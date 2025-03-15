from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

target_texts = ["This is sample.", "This is a pen."]

vectorstore = Chroma.from_texts(
    texts = target_texts,
    embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
)
docs = vectorstore.similarity_search_with_score('This is a pencile')

for doc in docs:
    print(doc)

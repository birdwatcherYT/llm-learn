from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# テキストファイルからドキュメントを作る
loader = TextLoader("./data/sample.txt")
text_splitter = CharacterTextSplitter(
    separator="\n\n", # 改行2つで区切ってチャンクを作る
    chunk_size=200, # チャンクの最大サイズ
    chunk_overlap=0, # チャンク同士の重なりの許容量
    length_function=len,
)
docs = loader.load_and_split(text_splitter)
vectorstore = Chroma.from_documents(
    docs, GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
prompt = hub.pull("langchain-ai/retrieval-qa-chat") # 中身はこちら https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(
    vectorstore.as_retriever(search_kwargs={"k": 1}), combine_docs_chain
)
print(rag_chain.invoke({"input": "What are the characteristics of the Oriental Turtle Dove?"}))
print(rag_chain.invoke({"input": "Tell me the pattern of the rock dove?"}))

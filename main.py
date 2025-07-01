import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA




from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


st.title("NewsBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")


urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

embedding_model = OpenAIEmbeddings(model ="text-embedding-3-large")


if process_url_clicked and urls:
 with st.spinner("Loading and processing articles..."):
  loader = UnstructuredURLLoader(urls=urls)
  data = loader.load()
  
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50) 
  chunks = text_splitter.split_documents(data)
  texts = [doc.page_content for doc in chunks]


 #print(chunks[0].page_content[:300])


  embedding_model = OpenAIEmbeddings(model ="text-embedding-3-large")
  vectors = embedding_model.embed_documents(texts)

  vector_store = FAISS.from_documents(chunks, embedding_model)

#print("\n--- Sample Embeddings ---\n")
#for i, vec in enumerate(vectors[:2]):
    #print(f"Embedding {i+1}: {vec[:5]}... (length={len(vec)})")



  vector_store.save_local("faiss_index")


st.success("Articles processed and indexed!")


vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)



st.markdown("---")
user_query = st.text_input("Enter your question: ")

def query_prompt(user_query: str, vector_store, embedding_model, k=3) -> str:
    """
    Builds a prompt by retrieving relevant context for a user query using FAISS.

    Args:
        user_query (str): The question or query from the user.
        vector_store (FAISS): The FAISS vector store.
        embedding_model (OpenAIEmbeddings): Your embedding model.
        k (int): Number of top similar documents to fetch.

    Returns:
        str: A formatted prompt including context and the original question.
    """

    # Step 1: Embed query and search FAISS (automatically handled inside similarity_search)
    results = vector_store.similarity_search(user_query, k=k)

    # Step 2: Combine top-k contexts into one block of text
    context = "\n\n".join([doc.page_content for doc in results])

    # Step 3: Create the final prompt
    prompt = f"""You are a knowledgeable and reliable assistant trained to answer questions based on recent news articles.

Use the information provided in the context below to respond to the user's question. 
Only use facts from the context. Do not make up information or include outside knowledge.

If the context is insufficient to answer the question, respond with "The provided articles do not contain enough information to answer that."

Answer clearly and concisely.

Context:
{context}

Question: {user_query}
Answer:"""

    return prompt



final_prompt = query_prompt(user_query, vector_store, embedding_model)
print(final_prompt)


if st.button("Get Answer") and user_query:
   with st.spinner("Thinking..."):
    results = vector_store.similarity_search(user_query, k=2)
   #for i, doc in enumerate(results):
    #print(f"\n--- Match {i+1} ---\n{doc.page_content}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8, max_tokens=500)

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" means all context is stuffed into a single prompt
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
)

    query = user_query
    result = qa_chain(query)
    st.markdown("### ✅ Answer")

    st.write(result["result"])
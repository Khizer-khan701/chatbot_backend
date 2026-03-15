from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from fastapi import HTTPException
from langchain_core.messages import AIMessage,HumanMessage
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
load_dotenv()
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
def load_vectorstore():
    try:
        pc=Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        vector_store=PineconeVectorStore(
            index_name=os.environ["PINECONE_INDEX_NAME"],
            embedding=embeddings,
            namespace="chatbot-docs"
        )
        return vector_store
    except Exception as e:
        raise HTTPException(status_code=400,detail={"message":str(e)})
def build_conversational_chain(vector_store):
    try:
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
        )
        model=ChatOpenAI(model='gpt-4')
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        create_history_retriever=create_history_aware_retriever(
            model,retriever,contextualize_q_prompt
        )
        
        system_prompt="""
    you are helpful python assistant.
    Read The given context carefully and then give the response according to the query and the given response in just 4 to 5 lines.
    Do not use the out information or do not give the response from any guess only give the answer from the context.
    Give the code examples if you find in the context.\n
    
    Context:
    {context}
    """
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        document_chain=create_stuff_documents_chain(model,qa_prompt)
        return create_retrieval_chain(create_history_retriever,document_chain)
        
    except Exception as e:
        raise HTTPException(status_code=400,detail={"message":str(e)})
    

def run_rag(query,chat_history:None,chain:None):
    try:
        if chat_history is None:
            vector_store=load_vectorstore()
            chain=build_conversational_chain(vector_store)
        if chat_history is None:
            chat_history=[]
        chatbot_messages=[]
        for msg in chat_history:
            if msg.get("role")=="user":
                chatbot_messages.append(HumanMessage(content=msg.get("content","")))
            if msg.get("role")=="assistant":
                chatbot_messages.append(AIMessage(content=msg.get('content',"")))
        response=chain.invoke({"input":query,"chat_history":chatbot_messages})
        answer=response["answer"]
        return answer,chain
    except Exception as e:
        return HTTPException(status_code=400,detail={"message":str(e)})
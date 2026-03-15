from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rag_pipeline import run_rag
from fastapi.requests import Request
from rag_pipeline import build_conversational_chain,load_vectorstore

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(
    SessionMiddleware,
    secret_key="abc"
)

class Chatbot(BaseModel):
    message: str

@app.on_event("startup")
def startup_event():
    app.state.vectorstores={}
    app.state.rag_chains={}

def get_chain():
    vector_store=load_vectorstore()
    chain=build_conversational_chain(vector_store)
    app.state.vectorstores=vector_store
    app.state.rag_chains=chain
    return chain

@app.post("/chatbot")
def chatbot_api(payload:Chatbot,request:Request):
    message=(payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400,detail="User Query Not Found")
    chat_history=request.session.get("chat_history",[])
    chain=get_chain()
    result,_=run_rag(message,chat_history,chain)
    chat_history.append({"role":"user","content":message})
    chat_history.append({"role":"assistant","content":result})
    if len(chat_history)>20:
        chat_history=chat_history[-20:]
    request.session["chat_history"]=chat_history
    return JSONResponse(status_code=200,content={"response":result})

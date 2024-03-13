from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# main.py
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
import csv
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL="gpt-4-0125-preview"

def crawl_namuwiki(topic):
    # 나무위키 주소 설정
    url = f"https://namu.wiki/w/{topic}"
    
    # HTTP 요청 보내기
    response = requests.get(url)
    
    if response.status_code == 200:
        # HTML 파싱
        soup = BeautifulSoup(response.content, "html.parser")
        
        # 필요한 내용 추출
        content_tag = soup.find("div", {"class": "wiki-content"})
        if content_tag:
            content = content_tag.text.strip()
            return content
        else:
            return "해당 주제의 내용을 찾을 수 없습니다."
    else:
        return "페이지를 찾을 수 없습니다."





class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

want_to = """너는 아래 내용을 기반으로 질의응답을 하는 로봇이야.
content
{}
"""

# 나무위키에서 특정 주제의 내용 크롤링
topic = "마이클_조던"  # 크롤링할 주제 설정
content = crawl_namuwiki(topic)


st.header("백엔드 스쿨/파이썬 2회차(9기)")
st.info("마이클 조던에 대해 알아볼 수 있는 Q&A 로봇입니다.")
st.error("마이클 조던에 대한 내용이 적용되어 있습니다.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="안녕하세요! 챗봇 Q&A 로봇입니다. 어떤 내용이 궁금하신가요?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(openai_api_key=API_KEY, streaming=True, callbacks=[stream_handler], model_name=MODEL)
        response = llm([ ChatMessage(role="system", content=want_to.format(content))]+st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
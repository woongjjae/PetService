import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

from langchain.retrievers.multi_query import MultiQueryRetriever  
from langchain.prompts import ChatPromptTemplate

import folium
from streamlit_folium import folium_static
from streamlit_folium import st_folium
from streamlit_folium import folium_static

def main():
    #탭 최상단 이름, 아이콘
    st.set_page_config(
        page_title="PET CHAT",
        page_icon=":dog:")
    
    #페이지 제목
    st.title(":dog: :red[반려동물 동반 가능] 시설 챗봇")
    
    # session_state에 지정해주기
    # 이후에 사용해주기 위해 미리 지정
    if "conversation" not in st.session_state:  
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    
    #vector_db session_state에 저장
    if "database" not in st.session_state:  #database가 session_state에 없다면
        model_name='jhgan/ko-sroberta-nli' 
        model_kwargs = {"device":"cpu"}  
        encode_kwargs={'normalize_embeddings':True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        st.session_state.database = True
        st.session_state.vector_db = vector_db  #vector_db 새로 session_state에 저장 
    else:  #database가 session_state에 있다면
        vector_db = st.session_state.vector_db  # 이전에 저장된 vector_db 그대로 사용
    
    
    # 사이드 바
    #openai_api_key를 사이드바에서 입력하도록 한다.
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    
    # process 버튼을 누르면 구동되는 부분
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
            
        #session_state에 chain저장
        st.session_state.conversation = get_conversation_chain(vector_db,openai_api_key)
        st.session_state.processComplete = True
    
    #처음 인사말
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 저는 반려동물 동반 시설을 안내해주는 챗봇입니다. 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
    
    #채팅 인터페이스  
    #for문으로 메세지가 생성될 때마다 화면에 나타나도록
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  #역할에 따라 표시 (왼쪽 아이콘이라고 보면 됨)
            st.markdown(message["content"])  # 메세지 표시
    
    #메모리라고 보면 됨
    history = StreamlitChatMessageHistory(key="chat_messages")
    
    #사용자가 질문창에 질문을 입력하면,
    if query := st.chat_input("질문을 입력해주세요."):
        #session_state에 사용자 질문 입력
        st.session_state.messages.append({"role": "user", "content": query})
        
        #user 역할로 query의 내용를 표시
        with st.chat_message("user"):
            st.markdown(query)
            
        #assistant의 역할로 답변 화면에 나타내기
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            
            #로딩 표시
            with st.spinner("Thinking..."):
                #질문을 전달하여 chain호출, 결과는 result에 저장
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    #result의 chat history를 session_state에 저장
                    st.session_state.chat_history = result['chat_history']
                #result에서 answer부분만 response에 저장
                response = result['answer']
                #답변에 참고한 문서 정보를 source_document에 저장
                source_documents = result['source_documents']
                #assistant의 역할로 response 표시
                st.markdown(response)
                
                
                #지도 출력
                m = folium.Map(width=600, height=400, location=[37.5665, 126.9780], zoom_start=12)
                coordinates_found = False
                with st.expander("지도 확인"):  # 토글 속에 나타나도록
                    #for문으로 source_document의 모든 문서에 대해서
                    for doc in source_documents: 
                        metadata = doc.metadata
                        title = metadata["title"]
                        coordinates = metadata["coordinates"]
                        
                        #답변에 사용된 문서의 좌표만 지도에 나타내기 위한 if문
                        #답변에 title이 포함되어 있으면 marker 추가
                        if title in response:  
                            lat_str, lon_str = coordinates.split(", ")
                            lat = float(lat_str[1:])
                            lon = float(lon_str[1:])
                            folium.Marker([lat, lon], popup=title).add_to(m)
                            coordinates_found = True

                    if coordinates_found:  #coordinates_found = True 이면
                        folium_static(m)   #지도 출력
                    else:
                        st.markdown("표시할 위치가 없습니다") #지도 출력 x
                    
                    
        #session_state에 assistant(챗봇)의 답변 입력
        st.session_state.messages.append({"role": "assistant", "content": response})


def get_conversation_chain(vector_db,openai_api_key):
    
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o",temperature=0)
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(
        search_type='mmr',
        search_kwargs={'k':3, 'fetch_k': 3}),
        llm=llm
    )
    
    #prompt 설정
    prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful assistant. 
                    You are a chatbot that provides information about pet-friendly facilities. 
                    Please respond based on the context.
                    Answer questions using only the following context.
                    모든 정보를 알려줘. 간결하게 답해줘.
                    If you don't know the answer just say you don't know, 
                    don't make it up:
                    \n\n
                    {context}
                    """,
                ),
                ("human", "{question}"),
            ]
        )
    
    #langchain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,   #참고한 문서 출력
        verbose = True,  
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return conversation_chain
    
    
if __name__ == '__main__':
    main()
    

    
#anaconda prompt에서 streamlit run app.py 실행.
#사이드바에 openai api key를 입력한 뒤 process 버튼 클릭
#질문 시작하기
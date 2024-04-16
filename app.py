import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

def main():
    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant for chatting with PDF.")
        ]

    # load open_api_key
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()

      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)

      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)

      # show user input
      user_question = st.text_input("Ask a question about your PDF:", key="user_input")
      if user_question:
        st.session_state.messages.append(HumanMessage(content=user_question))
        docs = knowledge_base.similarity_search(user_question)

        # create OpenAI model
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with st.spinner("Thinking..."):
          response = chain.run(input_documents=docs, question=user_question)

        # save message history
        st.session_state.messages.append(
            AIMessage(content=response))

      # display message history
      messages = st.session_state.get('messages', [])
      for i, msg in enumerate(messages[1:]):
          if i % 2 == 0:
              message(msg.content, is_user=True, key=f'{str(i)}_user')
          else:
              message(msg.content, is_user=False, key=f'{str(i)}_ai')

if __name__ == '__main__':
    main()
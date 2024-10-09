import streamlit as st
from streamlit_chat import message
from backend.core import call_llm_with_rag


st.header("LangChain Documentation Helper")
prompt = st.text_input(
    "Prompt", placeholder="Enter your prompt here..."
)

if not "prompts" in st.session_state:
    st.session_state["prompts"] = []

if not "responses" in st.session_state:
    st.session_state["responses"] = []

if not "history" in st.session_state:
    st.session_state["history"] = []

if prompt:
    with st.spinner("Generating response..."):
        generated_response = call_llm_with_rag(
            query=prompt,
            vector_db_path="langchain-docs-faiss-vdb/",
            chat_history=st.session_state["history"],
        )
        sources = set(
            doc.metadata["source"]
            for doc in generated_response[
                "source_documents"
            ]
        )
        response = (
            f"{generated_response['result']}\n\n{sources}"
        )
        st.session_state["prompts"].append(prompt)
        st.session_state["responses"].append(response)
        st.session_state["history"].append(
            ("human", prompt)
        )
        st.session_state["history"].append(
            ("ai", generated_response["result"])
        )


if st.session_state["history"]:
    for response, query in zip(
        st.session_state["responses"],
        st.session_state["prompts"],
    ):
        message(query, is_user=True)
        message(response)

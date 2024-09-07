from LawyerAgents import LawAgent
import streamlit as st
import streamlit_chat as sc

def run():
    LA = LawAgent()
    response, legal_agent = LA.preprocessor() 
    if LA.query:
        st.session_state["chat_history"].append({"role": "user", "content": LA.query})
        st.chat_message("user").write(LA.query)
        st.session_state["chat_history"].append({"role": "assistant", "content": response["content"]})
        st.chat_message("assistant").write(response["content"])
    
    if query := st.chat_input("Say something"):
        response = LA.get_response(query=query)
        st.session_state["chat_history"].append({"role": "user", "content": query})
        st.chat_message("user").write(query)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.header(body="Legal Compliance Assistant", anchor=None, help=None, divider="rainbow")
    st.subheader("Ask your queries about the Company Law")
    run()
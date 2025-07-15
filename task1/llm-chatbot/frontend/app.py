import streamlit as st
import requests

# our backend endpoint
BACKEND_URL = "http://127.0.0.1:8500/chat"  

# our configuration
st.set_page_config(page_title="Adham GPT", layout="centered")
st.title("Adham GPT ")

# we initialize session state for conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# to display conversation history
for msg in st.session_state.messages:
    role = "You" if msg["role"] == "user" else "Adham GPT"
    st.markdown(f"**{role}:** {msg['content']}")

# User input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your message:")
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            with st.spinner("Generating response..."):
                response = requests.post(BACKEND_URL, json={"user_input": user_input})
                result = response.json()
                reply = result.get("response", "No response received.")
                elapsed = result.get("time", 0)

                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.success(f"Responded in {elapsed} seconds")

        except requests.exceptions.ConnectionError:
            st.error("Unable to connect to the backend.")
        except Exception as e:
            st.error(f"Error: {e}")

# Clearing chat button for new message 
if st.button("Clear chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# display full conversation history all of it under the button of the chat 
st.markdown("---")
st.subheader("Conversation History")

for msg in st.session_state.messages:
    role = "Adham GPT" if msg["role"] == "assistant" else "You"
    st.markdown(f"**{role}:** {msg['content']}")

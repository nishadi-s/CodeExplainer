import os
from dotenv import load_dotenv
import requests
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(page_title="Multilingual Code Explanation Assistant", layout="wide")

# Retrieve API key securely from environment variable
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # API key from environment variable
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Function to query the model for code explanation
def query_model(input_text, language):
    # Improved prompt to ensure structured responses
    prompt = (
        f"Explain the following code in {language}. Keep it simple and concise:\n\n"
        f"### Code:\n{input_text}\n\n"
        f"### Explanation (in {language}):"
    )

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 300, "temperature": 0.3, "top_p": 0.9},
    }

    # Send request to Hugging Face API
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        result = response.json()
        explanation = result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
        return explanation.strip() or "No explanation available."
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit sidebar content
st.sidebar.title("How to Use the App")
st.sidebar.markdown("""
1. Paste your code snippet into the input box.
2. Enter the language you want the explanation in (e.g., English, Spanish, French).
3. Click 'Generate Explanation' to see the results.
""")
st.sidebar.divider()
st.sidebar.markdown(
   """
   <div style="text-align: center;color: grey;">
       Made with ♡ by Ana
   </div>
   """,
   unsafe_allow_html=True
)

# Main content
st.title("Multilingual Code Explanation Assistant")
st.markdown("### Powered by Mistral-7B 🦙")

# Code input and language selection
code_snippet = st.text_area("Paste your code snippet here:", height=200)
preferred_language = st.text_input("Enter your preferred language for explanation (e.g., English, Spanish):")

# Generate explanation button
if st.button("Generate Explanation"):
    if code_snippet and preferred_language:
        with st.spinner("Generating explanation... ⏳"):
            explanation = query_model(code_snippet, preferred_language)
        st.subheader("Generated Explanation:")
        st.write(explanation)
    else:
        st.warning("⚠️ Please provide both the code snippet and preferred language.")

# Footer section
st.markdown("---")
st.markdown("🧠 **Note**: This app uses Mistral-7B from Hugging Face for multilingual code explanations.")

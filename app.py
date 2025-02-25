import streamlit as st
from ingest import ingest_urls
from rag import answer_query

st.title("Web Content Q&A Tool")

# User input for OpenAI API key
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

if openai_api_key:
    openai_api_key = openai_api_key.strip()
    # llm = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=openai_api_key)
    st.success("API Key Set Successfully!")
else:
    st.warning("Please enter your OpenAI API Key.")

# User inputs multiple URLs (comma-separated)
urls_input = st.text_area("Enter one or more URLs (comma-separated)")

# Button to start ingestion
if st.button("Ingest URLs"):
    urls = [url.strip() for url in urls_input.split(",") if url.strip()]
    if urls and openai_api_key:
        ingest_urls(urls, openai_api_key)
        st.success(f"Ingested {len(urls)} URLs successfully!")
    else:
        st.warning("Please enter at least one URL and your OpenAI API Key.")

# # URL Input
# url = st.text_input("Enter a URL to ingest:")
# if st.button("Ingest URL"):
#     if url:
#         with st.spinner("Ingesting content..."):
#             ingest_url(url, openai_api_key)
#         st.success("Content ingested successfully!")
#     else:
#         st.error("Please enter a valid URL.")

# Question Input
query = st.text_area("Ask a question based on the ingested content:")
if st.button("Get Answer"):
    if query:
        with st.spinner("Fetching answer..."):
            answer = answer_query(query, openai_api_key)
        st.write("**Answer:**", answer)
    else:
        st.error("Please enter a question.")

import streamlit as st 
import uuid
from utils import *

if "unique_id" not in st.session_state:
    st.session_state["unique_id"] = ""

def main():
    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance 🤖")
    st.subheader("I can help you with resume process")

    job_description = st.text_area("Enter the job description here ...", key = "1")
    job_count = st.text_input("No. of resumes to return", key="2")

    pdf = st.file_uploader("Upload resumes here, only pdf allowed", type=["pdf"], accept_multiple_files=True)
    submit = st.button("Submit")

    if submit:
        with st.spinner("Wait for it..."):
            st.write("Process starts")

            # Creating Unique ID
            st.session_state["unique_id"]=uuid.uuid4().hex
            st.write(st.session_state["unique_id"])

            # Creating Document List
            docs = create_docs(pdf, st.session_state["unique_id"])
            # st.write(docs)
            
            # Printing Total number of Documents
            st.write(len(docs))

            # Creating Embeddings
            embeddings = create_embeddings()

            # Pushing Data to pinecone
            push_pinecone(embeddings, docs)
            st.write("Push Done")

            # Pulling Data from pinecone
            vector_store = pull_pinecone(embeddings)
            st.write("Pull Done")

            # Relevant docs
            relevant_docs = similar_docs(vector_store, job_description, st.session_state["unique_id"], job_count)
            # st.write(relevant_docs)
            # st.write(len(relevant_docs))

            # Line Separator
            st.write(":heavy_minus_sign:" * 30)

            # Summary of resumes
            for item in range(len(relevant_docs)):
                st.subheader("➡️" + str(item+1))

                st.write("File : " + relevant_docs[item][0].metadata["name"])

                with st.expander("Show me"):
                    st.info("Match Score : " + str(relevant_docs[item][1]))
                    summary = get_summary(relevant_docs[item][0])
                    st.write("Summary : " + summary)


        st.success("All Done")

if __name__ == "__main__":
    main()


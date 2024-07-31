import boto3 
import streamlit as st
import os
from dotenv import load_dotenv,find_dotenv
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

load_dotenv(find_dotenv())

s3_client= boto3.client("s3")
BUCKET_NAME=os.getenv("BUCKET_NAME")
bedrock_client=boto3.client(service_name= "bedrock-runtime",
                            aws_access_key_id=os.getenv("ACESS_KEY_ID"),
                      aws_secret_access_key=os.getenv("ACCESS_SECRET"),
                      region_name=os.getenv("AWS_REGION"))

bedrock_embeddings= BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock_client)




def split_text(pages,chunk_size, chunk_overlap):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs= text_splitter.split_documents(pages)
    return docs



def create_vector_store(file_uploaded_name,documents):
    vector_store_faiss=FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{file_uploaded_name}.bin"
    folder_path= "/tmp/"
    vector_store_faiss.save_local(index_name=file_name,folder_path=folder_path)
    s3_client.upload_file(Filename= folder_path + "/" + file_name + ".faiss", Bucket= BUCKET_NAME, Key= 'my_faiss.faiss')
    s3_client.upload_file(Filename= folder_path + "/" + file_name + ".pkl", Bucket= BUCKET_NAME, Key= 'my_faiss.pkl')
    return True

def main():
    st.write("Teacher side streamlit interface")
    uploading_file= st.file_uploader("Upload a pdf file",type=["pdf"])

    if uploading_file is not None:
        saved_file= uploading_file.name
        with open(saved_file,"wb") as file:
            file.write(uploading_file.getvalue())

        loader= PyPDFLoader(saved_file)
        pages= loader.load_and_split()
        st.write(f"Total Pages {len(pages)}")

        splitted_docs= split_text(pages,1000,200)
        st.write(f"Total splitted docs {len(splitted_docs)}")

        st.write(f"Doc 1 {splitted_docs[0]}")
        st.write(f"Doc 2 {splitted_docs[1]}")

        result = create_vector_store(saved_file, splitted_docs)

        if result:
            st.write("Hurray!! PDF processed successfully")
        else:
            st.write("Error!! Please check logs.")

if __name__=="__main__":
    main()

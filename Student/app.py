import boto3 
import streamlit as st
import os
import uuid
from dotenv import load_dotenv,find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from botocore.exceptions import ClientError


load_dotenv(find_dotenv())

s3_client= boto3.client("s3")
BUCKET_NAME="inference-aws-1"
bedrock_client=boto3.client(service_name= "bedrock-runtime",aws_access_key_id=os.getenv("ACESS_KEY_ID"),
                      aws_secret_access_key=os.getenv("ACCESS_SECRET"),
                      region_name="us-west-2")
bedrock_embeddings= BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock_client)


folder_path= "/tmp/"
def load_index():
    try:
        s3_client.download_file(Bucket= BUCKET_NAME,Key="my_faiss.faiss",Filename=f"{folder_path}my_faiss.faiss")
        s3_client.download_file(Bucket= BUCKET_NAME,Key= f"my_faiss.pkl",Filename=f"{folder_path}my_faiss.pkl")

    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("The object does not exist.")
        else:
            print(f"Error occurred: {e}")

def get_response(llm, vectorstore,question):
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""
    PROMPT=PromptTemplate(input_variables=["context","question"],template=prompt_template)
    qa= RetrievalQA.from_chain_type(
        llm=llm,chain_type="stuff",
        retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT})
    answer= qa({"query":question})
    return answer['result']    

def get_llm():
    try:
        llm= ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0",client=bedrock_client)
        return llm
    except ClientError as e:
        print(f"Error initializing LLM: {e}")

def main():
    st.write("Student side streamlit interface")

    load_index()
    dir_list=os.listdir(folder_path)
    st.write(f"Files and folders in {folder_path}")
    st.write(f"{dir_list}")
    faiss_index=FAISS.load_local(index_name="my_faiss",folder_path=folder_path,embeddings=bedrock_embeddings,allow_dangerous_deserialization=True)
    st.write("Index is ready")
    question = st.text_input("Please ask your question")

    if st.button('ask question'):
        with st.spinner("Querying"):
            llm=get_llm()
            st.write(get_response(llm, faiss_index, question))
            st.success("Done")

if __name__=="__main__":
    main()

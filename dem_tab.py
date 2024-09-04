import streamlit as st
from openai import OpenAI
import numpy as np
import pickle
import json
import requests

ROOT_NAME = "Demo"
MODEL_NAME = "full_transcript_mlp_best_model_one"
MAPPER = {"0": "Normal", "1": "Alzheimer's" }
CLIENT = openapi(api_key=st.secrets["OPEN_API_KEY"])

def save_feedbacks(data):
    url = "https://s5bcsbu84l.execute-api.us-east-1.amazonaws.com/Research/record-history"
    r = requests.post(url, data=json.dumps(data))
    response = getattr(r,'_content').decode("utf-8")
    response = json.loads(response)
    print(response)
    saved = response["data"]["saved"]
    return saved

def create_payload_to_record(user_id, user_context, label):
    payload = {"httpMethod": "POST", "body": 
            {"rootName": ROOT_NAME,
            "userId": user_id,
            "userText": user_context,
            "prediction": label}}

    return payload

@st.cache_resource 
def load_model(path:str):
    with open(path, 'rb') as pickle_file:
         model_file = pickle.load(pickle_file)
    return model_file

def get_openapi_embeddings(text:str)-> list:
    response = CLIENT.embeddings.create(input=f"{text}", model= "text-embedding-ada-002")
    return response.data[0].embedding

def alzheimers_dem_app():
    authorization_response = st.query_params
    final_model = load_model(MODEL_NAME)
    if "code" in authorization_response:
        st.header("Alzheimer's Tracker")
        text_input = st.text_input("Please enter text here...")
        if text_input:
            if st.button("Make prediction"):
                embeddings = get_openapi_embeddings(text_input)
                prd = final_model.predict(np.array([embeddings]))
                prd_label = MAPPER[str(prd[0])]
                set_payload = create_payload_to_record(st.session_state.user_id, text_input, prd_label)
                saved_db = save_feedbacks(set_payload)
                if saved_db:
                    st.toast("Record saved.")
                    st.subheader("Your Condition is {}".format(prd_label)")
                else:
                    st.error("Record saved failed.")
    else:
        st.error("Please sign in first.")

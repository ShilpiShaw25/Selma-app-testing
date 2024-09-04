import streamlit as st
import json
import requests
import pandas as pd

ROOT_NAME = "kalemvesilgi"

def get_feedbacks(data):
    url = "https://s5bcsbu84l.execute-api.us-east-1.amazonaws.com/Research/record-history"
    r = requests.get(url, data=json.dumps(data))
    response = getattr(r,'_content').decode("utf-8")
    response = json.loads(response)
    records = response["data"]["records"]
    return records

def payload_creator(user_id):
    get_payload = {"httpMethod": "GET", "queryStringParameters": {"rootName": ROOT_NAME, "userid": user_id}}

    return get_payload


def history_tab_ui():
    st.header("Hello,{}".format(st.session_state.user_name))
    st.subheader("Here are your past records!")
    payload = payload_creator(st.session_state.user_id)
    records = get_feedbacks(payload)

    df = pd.DataFrame(records)
    df.drop("Id", axis=1, inplace=True)
    st.dataframe(df)
    

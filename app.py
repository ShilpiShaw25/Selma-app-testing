import streamlit as st
import json
import requests
from requests_oauthlib import OAuth2Session
from dem_tab import alzheimers_app
from history_tab import history_tab_ui

TITLE = "Alzheimer's App"
IMAGE_CAPTION = "Let's track Alzheimer's!"
CSS_STYLE = """ 
<style>
.login-button {
            padding: 10px;
            border-radius: 5px;
            background-color: blue;
            color: white;
            border: none;
            margin-bottom: 20px;
   }

.login-button: hover {
            color: black;
        }
</style>
"""

LOGIN_IMAGE = "https://kffhealthnews.org/wp-content/uploads/sites/2/2020/02/Dementia-resized.png?w=1024"
SIGN_IN_IMAGE = "https://beconnected.esafety.gov.au/pluginfile.php/82020/mod_resource/content/1/t35_c4_a2_p1.png"
LOG_OUT_URL = "https://fxn4qbs5wygh8lzfmgfpuy.streamlit.app"

st.markdown(CSS_STYLE, unsafe_allow_html=True)

client_id = st.secrets["CLIENT_ID"]
client_secret = st.secrets["CLIENT_SECRET"]
redirect_uri = "https://fxn4qbs5wygh8lzfmgfpuy.streamlit.app/"
authorization_based_url = "https://accounts.google.com/o/oauth2/auth"
token_url = "https://oauth2.googleapis.com/token"
scope = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

def exchange_code_for_token(code):
            token_url = "https://oauth2.googleapis.com/token"
            data = {
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                }
            response = requests.post(token_url, data=data)
            response_data = response.json()

            if response_data != 200:
                        raise Exception("Failed to retrieve token: " + response_data.get("error_description", ""))

            return response_data["access_token"]


def get_user_info(access_token):
            user_info_url = "https://www.googleapis.com/oauth2/v3/userinfo"
            headers = {"Authorization": f"Bearer {access_token}"}
            
            response = requests.get(user_info_url, headers=headers)
            user_info = response.json()

            if response.status_code != 200:
                        raise Exception("Failed to retrieve user info: " + user_info.get("error_description", ""))
                        
            return user_info


if "oauth_state" not in st.session_state:
    st.session_state.oauth_state = None
if "oauth_token" not in st.session_state:
    st.session_state.oauth_token = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None

if st.session_state.oauth_token:
    with st.sidebar:
        st.title("Log Out")
        st.image(SIGNIN_IMAGE, caption="Log Out")
        if st.button("Log Out"):
            st.session_state.oauth_token = None
            # set login button
            st.write(
                f"""
            <a target="_self" href="{LOGOUT_URL}">
                <button class = 'login-button'>
                    Confirm
                </button>
            </a>
            """,
                unsafe_allow_html=True,
            )

    # set title and image
    st.title(TITLE)
    st.image(LOGIN_IMAGE, caption=IMAGE_CAPTION)
    dementia_app_tracker, history = st.tabs(["Dementia Checker", "Get History"])

    with alzheimer_app_tracker:
        dementia_app()

    with history:
        history_tab()
else:
    oauth2_session = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scope)
    authorization_url, state = oauth2_session.authorization_url(
        authorization_base_url, access_type="offline")

    if st.session_state.oauth_state is None:
        st.session_state.oauth_state = state

    # set the sign in page
    st.title(TITLE)
    st.image(LOGIN_IMAGE, caption = "Let's Track Dementia")
    # sign in
    st.subheader("Please Sign In with Google to Continue 📲")
    with st.sidebar:
        st.subheader("Please Sign In")
        st.image(SIGNIN_IMAGE, caption = "Sign In")
        st.write(
            f"""
        <a target="_blank" href="{authorization_url}">
            <button class = 'login-button'>
                Google Sign In
            </button>
        </a>
        """,
            unsafe_allow_html=True,
        )
    authorization_response = st.query_params

    if "code" in authorization_response:
        st.session_state.oauth_token = exchange_code_for_token(
            authorization_response["code"]
        )
        user_info = get_user_info(st.session_state.oauth_token)
        # set session states
        st.session_state.user_id = user_info["sub"]
        st.session_state.user_name = user_info["name"]

        # ui
        with st.sidebar:
            st.subheader("Hi! {} 👋".format(user_info["name"]))
            st.subheader("Welcome to the Alzheimer's App!")
            st.write("**Name**: {}".format(user_info["name"]))
            st.write("**Email**: {}".format(user_info["email"]))

            st.subheader("Login Successful! Please Click on Continue")
            if st.button("Continue"):
                st.toast("Login Successfull!")

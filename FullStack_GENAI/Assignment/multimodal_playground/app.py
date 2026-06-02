import streamlit as st

from utils.router import route_text

from modalities.text import TextModality
from modalities.image import ImageModality
from modalities.audio import AudioModality
from modalities.video import VideoModality


st.set_page_config(page_title="Multimodal Playground",page_icon="🚀",layout="wide")

###################################################
# CUSTOM CSS
###################################################

st.markdown("""

<style>

.stApp{

background:
linear-gradient(
135deg,
#0F172A 0%,
#111827 50%,
#1E293B 100%
);

color:white;

}


/* Main Title */

.big-title{

font-size:42px;

font-weight:bold;

background:
linear-gradient(
90deg,
#00DBDE,
#FC00FF
);

-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

text-align:center;

padding-bottom:20px;

}


/* Cards */

.card{

background:#1E293B;

padding:25px;

border-radius:20px;

box-shadow:
0px 5px 25px rgba(
0,
0,
0,
0.4
);

}


/* Section Headers */

h1,h2,h3{

color:#FFFFFF !important;

}


/* Labels */

label{

color:#E5E7EB !important;

font-weight:600 !important;

font-size:16px !important;

}


/* Selectbox Text */

.stSelectbox div[data-baseweb="select"]{

background:#111827;

color:white;

border-radius:12px;

}

.stSelectbox div{

color:white !important;

}


/* Text Area */

.stTextArea textarea{

background:#111827;

color:white !important;

border-radius:15px;

font-size:16px;

}


/* Placeholder Text */

textarea::placeholder{

color:#9CA3AF !important;

}


/* Button */

.stButton > button{

width:100%;

height:55px;

font-size:20px;

font-weight:bold;

border:none;

border-radius:15px;

background:
linear-gradient(
90deg,
#00DBDE,
#FC00FF
);

color:white;

}


/* Output Card */

.output-card{

background:#111827;

padding:25px;

border-radius:20px;

border:1px solid #374151;

color:white;

}


/* Markdown Text */

p{

color:white !important;

}


/* Sidebar / widget text */

[data-testid="stMarkdownContainer"]{

color:white;

}

</style>

""", unsafe_allow_html=True)



###################################################
# TITLE
###################################################

st.markdown(
'<div class="big-title">🚀 Multimodal Playground</div>',
unsafe_allow_html=True
)

left,right=st.columns([1,3])


###########################################
# LEFT PANEL
###########################################

with left:

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h2 style='color:white'>⚙ Configuration</h2>",
        unsafe_allow_html=True
    )

    provider=st.selectbox(

        "Provider",

        [

            "groq",

            "openai",

            "gemini",

            "claude"

        ]

    )


    modality=st.selectbox(

        "Modality",

        [

            "Text → Text",

            "Text → Image",

            "Image → Text",

            "Text → Audio",

            "Audio → Text",

            "Text → Video",

            "Video → Text"

        ]

    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


###########################################
# RIGHT PANEL
###########################################

uploaded_file=None

with right:

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h2 style='color:white'>💬 Prompt</h2>",
        unsafe_allow_html=True
    )

    prompt=st.text_area(

        "Enter Prompt",

        height=200

    )


    ###################################
    # Dynamic Upload Components
    ###################################

    if modality=="Image → Text":

        uploaded_file=st.file_uploader(

            "Upload Image",

            type=["png","jpg","jpeg"]

        )


    elif modality=="Audio → Text":

        uploaded_file=st.file_uploader(

            "Upload Audio",

            type=["mp3","wav"]

        )


    elif modality=="Video → Text":

        uploaded_file=st.file_uploader(

            "Upload Video",

            type=["mp4","mov"]

        )


    generate=st.button(
        "Generate ✨"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


###########################################
# GENERATION
###########################################

if generate:

    provider_obj=route_text(
        provider
    )

    response=""


    ###################################
    # TEXT MODALITIES
    ###################################

    text_model=TextModality(
        provider_obj
    )

    if modality=="Text → Text":

        response=text_model.text_to_text(
            prompt
        )


    elif modality=="Text → Image":

        response=text_model.text_to_image(
            prompt
        )


    elif modality=="Text → Audio":

        response=text_model.text_to_audio(
            prompt
        )


    elif modality=="Text → Video":

        response=text_model.text_to_video(
            prompt
        )


    ###################################
    # IMAGE
    ###################################

    elif modality=="Image → Text":

        if uploaded_file is None:

            st.warning(
                "Upload image first"
            )

            st.stop()


        image_model=ImageModality(
            provider_obj
        )

        response=image_model.image_to_text(

            uploaded_file.read(),

            prompt

        )


    ###################################
    # AUDIO
    ###################################

    elif modality=="Audio → Text":

        if uploaded_file is None:

            st.warning(
                "Upload audio"
            )

            st.stop()


        audio_model=AudioModality(
            provider_obj
        )

        response=audio_model.audio_to_text(

            uploaded_file.read()

        )


    ###################################
    # VIDEO
    ###################################

    elif modality=="Video → Text":

        if uploaded_file is None:

            st.warning(
                "Upload video"
            )

            st.stop()


        video_model=VideoModality(
            provider_obj
        )

        response=video_model.video_to_text(

            uploaded_file.read(),

            prompt

        )


    ###################################
    # OUTPUT
    ###################################

    st.markdown(
    """
    <div class='output-card'>
    <h2>🧠 Output</h2>
    """,
    unsafe_allow_html=True
    )

    st.write(response)

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )
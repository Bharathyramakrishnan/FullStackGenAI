import streamlit as st
from PIL import Image
import tempfile
import io

from providers.text_models import text_response
from providers.image_models import generate_image,image_to_text
from providers.audio_models import speech_to_text
from providers.video_models import extract_frame
from utils.audio_utils import save_uploaded_audio
from utils.video_utils import save_uploaded_video,extract_frame,video_information


########################################

st.set_page_config(

page_title="Multimodal Playground",

layout="wide",

page_icon="🚀"

)

########################################

st.markdown("""

<style>

.main{

background:
linear-gradient(
135deg,
#0f172a,
#1e293b,
#2563eb
);

}

section[data-testid="stSidebar"]{

background:
linear-gradient(
180deg,
#111827,
#1e40af
);

}

.stButton button{

width:100%;

height:50px;

border-radius:15px;

font-size:18px;

font-weight:bold;

background:#ff4b4b;

color:white;

}

.outputbox{

padding:20px;

border-radius:15px;

background:#111827;

}

</style>

""",

unsafe_allow_html=True)

########################################

st.title(
"🚀 Multimodal AI Playground"
)

left,right=st.columns([1,3])

########################################
# LEFT PANE
########################################

with left:

    st.header("Settings")

    provider=st.selectbox(

        "Provider",

        [

        "Groq",

        "HuggingFace",

        "Local"

        ]

    )

    modality_from=st.selectbox(

        "From",

        [

        "Text",

        "Image",

        "Audio",

        "Video"

        ]

    )

    modality_to=st.selectbox(

        "To",

        [

        "Text",

        "Image"

        ]

    )

########################################
# RIGHT PANE
########################################

with right:

    st.header(
    "Input"
    )

    uploaded=None
    prompt=None

    if modality_from=="Text":

        prompt=st.text_area(

        "Enter Prompt",

        height=200

        )

    else:

        uploaded=st.file_uploader(

            f"Upload {modality_from}",

            type=None

        )

########################################

    if st.button(
    "Generate"
    ):

        with st.spinner(
        "Processing..."
        ):

            output=None

########################
# TEXT → TEXT
########################

            if modality_from=="Text" \
            and modality_to=="Text":

                output=text_response(
                    prompt
                )

########################
# TEXT → IMAGE
########################

            elif modality_from=="Text" \
            and modality_to=="Image":

                image=generate_image(prompt)
                st.image(image,use_container_width=True)


########################
# IMAGE → TEXT
########################

            elif modality_from=="Image":

                with tempfile.NamedTemporaryFile(

                delete=False,

                suffix=".jpg"

                ) as tmp:

                    tmp.write(
                        uploaded.read()
                    )

                    path=tmp.name

                output=image_to_text(
                    path
                )

########################
# AUDIO → TEXT
########################

            elif modality_from=="Audio":

                audio_path=save_uploaded_audio(uploaded) 
               # wav_path=convert_to_wav(audio_path)
               # duration=audio_duration(wav_path)
              #  st.info(f"Audio Duration: {duration} sec")
                output=speech_to_text(audio_path)

########################
# VIDEO → TEXT
########################

            elif modality_from=="Video":

                video_path=save_uploaded_video(uploaded)
                video_information=video_information(video_path)
                frame=extract_frame(video_path)
                output=image_to_text(frame)

########################

            if output:

                st.markdown(

                '<div class="outputbox">',

                unsafe_allow_html=True

                )

                st.subheader(
                "Output"
                )

                st.write(
                output
                )

                st.markdown(
                "</div>",
                unsafe_allow_html=True
                )
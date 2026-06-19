from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from groq import Groq

import os
import base64

load_dotenv()

#####################################
# HuggingFace Client
#####################################

hf_client=InferenceClient(

token=os.getenv(
"HUGGINGFACEHUB_API_TOKEN"
)

)

#####################################
# Groq Client
#####################################

groq_client=Groq(

api_key=os.getenv(
"GROQ_API_KEY"
)

)

#####################################
# TEXT -> IMAGE
#####################################

def generate_image(prompt):

    image=hf_client.text_to_image(

        prompt,

        model="stabilityai/stable-diffusion-xl-base-1.0"

    )

    return image


#####################################
# IMAGE -> TEXT
#####################################

def image_to_text(image_path):

    with open(
        image_path,
        "rb"
    ) as f:

        encoded=base64.b64encode(

            f.read()

        ).decode()

    completion=groq_client.chat.completions.create(

        model="meta-llama/llama-4-scout-17b-16e-instruct",

        messages=[

            {

            "role":"user",

            "content":[

                {

                "type":"text",

                "text":"Describe this image"

                },

                {

                "type":"image_url",

                "image_url":{

                "url":

f"data:image/jpeg;base64,{encoded}"

                }

                }

            ]

            }

        ]

    )

    return completion.choices[0].message.content
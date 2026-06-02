import requests
import os


HUGGINGFACEHUB_API_TOKEN=os.getenv(
    "HUGGINGFACEHUB_API_TOKEN"
)


def text_to_image(

    self,

    prompt

):

    API_URL=(
    "https://api-inference.huggingface.co/models/"
    "stabilityai/stable-diffusion-xl-base-1.0"
    )



    headers={

        "Authorization":

        f"Bearer {HUGGINGFACEHUB_API_TOKEN}"

    }



    response=requests.post(

        API_URL,

        headers=headers,

        json={

            "inputs":prompt

        }

    )



    if response.status_code!=200:

        return (

            "Image generation failed: "

            + response.text

        )



    return response.content
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

client=Groq(
api_key=os.getenv("GROQ_API_KEY")
)

def text_response(prompt):

    response=client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[

            {
                "role":"user",
                "content":prompt
            }

        ]

    )

    return response.choices[0].message.content
import tempfile
import os


def save_uploaded_audio(uploaded):

    suffix=os.path.splitext(
        uploaded.name
    )[1]

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=suffix

    ) as tmp:

        tmp.write(
            uploaded.read()
        )

        return tmp.name
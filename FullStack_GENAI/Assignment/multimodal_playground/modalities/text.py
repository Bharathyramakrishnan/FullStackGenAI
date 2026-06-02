class TextModality:

    def __init__(self,provider):

        self.provider = provider


    ####################################
    # TEXT → TEXT
    ####################################

    def text_to_text(

        self,

        prompt

    ):

        return self.provider.text_to_text(
            prompt
        )


    ####################################
    # TEXT → IMAGE
    ####################################

    def text_to_image(

        self,

        prompt

    ):

        return self.provider.text_to_image(
            prompt
        )


    ####################################
    # TEXT → AUDIO
    ####################################

    def text_to_audio(

        self,

        prompt

    ):

        return self.provider.text_to_audio(
            prompt
        )


    ####################################
    # TEXT → VIDEO
    ####################################

    def text_to_video(

        self,

        prompt

    ):

        return self.provider.text_to_video(
            prompt
        )
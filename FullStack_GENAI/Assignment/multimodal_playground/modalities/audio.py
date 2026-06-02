class AudioModality:

    def __init__(self,provider):

        self.provider=provider


    def audio_to_text(

        self,

        audio

    ):

        return self.provider.audio_to_text(
            audio
        )
class VideoModality:

    def __init__(self,provider):

        self.provider=provider


    def video_to_text(

        self,

        video,

        prompt

    ):

        return self.provider.video_to_text(

            video,

            prompt

        )
class ImageModality:

    def __init__(

        self,

        provider

    ):

        self.provider=provider


    def image_to_text(self,image,prompt):

        return self.provider.image_to_text(image,prompt)
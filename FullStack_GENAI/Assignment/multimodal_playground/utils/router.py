from providers.groq_provider import GroqProvider


def route_text(provider):

    if provider=="groq":

        return GroqProvider()

    raise Exception(
        "Provider not supported"
    )
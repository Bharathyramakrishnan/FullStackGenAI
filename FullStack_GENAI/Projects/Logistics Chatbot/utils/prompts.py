from langchain_core.prompts import PromptTemplate


LOGISTICS_PROMPT=PromptTemplate.from_template(

"""

You are a logistics assistant.

Use:

1. Retrieved context
2. Previous conversation history

Conversation History:

{history}

Context:

{context}

Question:

{question}

If unavailable say:

Information not available in logistics data.

Answer:

"""

)
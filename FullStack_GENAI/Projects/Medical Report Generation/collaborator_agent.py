import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompts import COLLABORATOR_PROMPT, ERROR_PROMPT

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
class CollaboratorAgent:
    def __init__(self):
        self.client = ChatGroq()

    def generate_section(self, section, context):
        """
        Generate a specific section of the medical report based on context
        
        Args:
            section (str): The section to generate (e.g. "Patient_history")
            context (str): Relevant context from the transcript for this section
        Returns:
            str: Generated content for the specified section
        """
        try:
            # Format the prompt with the section and context
            prompt = COLLABORATOR_PROMPT.format(section=section, context=context)   
            # Get generated content from GPT
            response = self.client.chat.compliation.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system","content": prompt}
                ],
                temperature=0.3
            )
            # Extract the generated content
            generated_content = response.choices[0].message.content

            return generated_content
        except Exception as e:
            print(f"Error generating report section: {e}")
            return ERROR_PROMPT.format(error=str(e))
        
    def validate_section(self, section_content):
        """
        Validate the generated section content for medical accuracy and completeness

        Args:
            section_content (str): The generated content for a report section
        Returns:
            bool: True if the section is valid, False otherwise

        """
        if not section_content:
            return ValueError("Section content is empty")
        if len(section_content) < 10:
            return ValueError("Section content is too short to be valid")
        
        return True 
    
    def format_section(self, section_content):
        """
        Format the section content to ensure proper structure and readability

        Args:
            section_name (str): The name of the section being formatted
            section_content (str): The generated content for a report section

        Returns:
            str: Formatted section content
        """


        return f"""
        {section_name}
        {'='*len(section_name)}

        {section_content}
        """
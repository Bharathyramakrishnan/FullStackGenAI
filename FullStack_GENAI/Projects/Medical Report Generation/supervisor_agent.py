import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompts import SUPERVISOR_PROMPT
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

class SupervisorAgent: 
    def __init__(self):
        self.client = ChatGroq()
        self.sections = [
            "Patient_history", 
            "Diagnosis Summary", 
            "Treatment Plan",
              "Follow-up Recommendations"]

    def analyze_transcript(self, transcript):
        """
        Analyze transcript and break it down into sections
        
        Args:
            transcript (str): Raw transript script
        Returns:
            dict: Structured breakdown of sections
        """
        try:
            # Get initial analysis from GPT
            response = self.client.chat.compliation.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system","content": SUPERVISOR_PROMPT},
                    {"role": "system","content": transcript}
                ],
                temperature=0.3
            )
            # Extract the analysis
            analysis = response.choices[0].message.content

            # Structure the analysis into sections
            structured_data = self.structure_analysis(analysis)

            return structured_data
        except Exception as e:
            print(f"Error analyzing transcript: {e}")
            return None
    
    def _structure_analysis(self, analysis):
        """
        Structure the analysis into defined sections
        
        Args:
            analysis (str): Raw analysis from text
        Returns:
            dict: Structured breakdown of sections
        """
        structured_data = {}
        current_section = None
        current_content = []

        # split analysis into lines
        lines = analysis.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line indicates a section header
            if any(section in line for section in self.sections):
                # Save previous section content if exists
                if current_section and current_content:
                    structured_data[current_section] = ' '.join(current_content).strip()
                    current_content = []
                
                # Set new current section
                current_section=next(section for section in self.sections if section in line)
                current_content = []
            else:
                # Add line to current section content
                if current_section:
                    current_content.append(line)
            
        # Save last section content if exists
        if current_section and current_content:
            structured_data[current_section] = ' '.join(current_content)
        return structured_data
    
    def validate_analysis(self, structured_data):
        """
        Validate the structured analysis for completeness and accuracy
        
        Args:
            structured_data (dict): Structured breakdown of sections
        Returns:
            bool: True if valid, False otherwise
        """
        missing_sections = [section for section in self.sections if section not in structured_data]
        
        if missing_sections:
            raise ValueError(f"Missing sections: {', '.join(missing_sections)}")
        
        return True

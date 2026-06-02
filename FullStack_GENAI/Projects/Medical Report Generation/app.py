import streamlit as st
import os
from dotenv import load_dotenv
from transriber import Transcriber
from supervisor_agent import SupervisorAgent
from collaborator_agent import CollaboratorAgent
from prompts import ERROR_PROMPT

load_dotenv()

# Initialize agents and transcriber
transcriber = Transcriber()
supervisor_agent = SupervisorAgent()
collaborator_agent = CollaboratorAgent()

def main():
    st.title("AI Medical Report Generator Assistant")
    st.write("Upload doctor's audio notes to generate a structural medical report.")

    # File upload
    audio_file = st.file_uploader("Upload Doctor's Audio Notes (mp3 or wav)", type=["mp3", "wav"])

    # Text Input for manual transcript
    transcript_input = st.text_area("Or paste the medical transcript here",height=200)

    if st.button("Generate Medical Report"):
        try:
            # get Transcript
            if audio_file:
                # validate audio file
                transcriber.validate_audio(audio_file)

                # show processing message
                with st.spinner("Transcribing audio..."):
                    transcript = transcriber.transcribe(audio_file)
            elif transcript_text:
                transcript = transcript_text
            else:
                st.error("Please upload an audio file or provide a transcript.")
                return
            
            # Display transcript
            st.subheader("Medical Transcript")
            st.write(transcript)

            # Analyze with Script
            with st.spinner("Analyzing transcript and generating report..."):
                structured_analysis = supervisor.analyze_transcript(transcript)
                supervisor.validate_analysis(structured_analysis)

            # Generate report sections 
            with st.spinner("Generating medical report sections..."):
                final_report = []

                for section,context in structured_analysis.items():
                    # Generate section content 
                    section_content = collaborator_agent.generate_section(section, context)
                    collaborator_agent.validate_section(section_content)

                    # format section for final report
                    formatted_section = collaborator_agent.format_section(section, section_content)
                    final_report.append(formatted_section)

            # Display final report
            st.subheader("Generated Medical Report")
            st.write("\n\n".join(final_report))

            # Add download option for report
            st.download_button(
                label="Download Medical Report",
                data=report_text,
                file_name="medical_report.txt",
                mime="text/plain"
            )   
        except Exception as e:
            st.write(ERROR_PROMPT.format(error=str(e)))
            
if __name__ == "__main__":
    main()

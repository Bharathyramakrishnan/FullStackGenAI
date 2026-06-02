import os
import whisper
from dotenv import load_dotenv
import tempfile
from pydub import AudioSegment

load_dotenv()

class Transcriber:
    def __init__(self):
        self.model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
    
    def convert_to_wav(self, audio_file):
        """ convert audio file to wav format if it's not already in wav format """
        if audio_file.name.endswith('.wav'):
            return audio_file.name
        # create temporary wav file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file_path = temp_file.name
        temp_file.close()

        #convert to wav
        audio = AudioSegment.from_file(audio_file)
        audio.export(temp_file_path, format="wav")

        return temp_file.name
    
    def transcribe(self, audio_file):
        """ transcribe audio file to text using whisper model.
        Args:
            audio_file: File object (mp3 or wav)
        Returns:
            str: Transcribed text
        """
        try:
            # convert to wav if needed 
            wav_file_path = self.convert_to_wav(audio_file)
            # Transcribe
            result = self.model.transcribe(wav_file_path)
            #clean up temporary wav file if it was created
            if wav_file_path != audio_file.name:
                os.unlink(wav_file_path)  # delete temporary wav file
       
            return result["text"]
        except Exception as e:
            raise Exception(f'Error during transcription: {str(e)}')
        
    def validate_audio(self, audio_file):
        """ validate audio file format and size.
        Args:
            audio_file: File object 
        Returns:
            bool: True if valid, False otherwise
        """
        if not audio_file:
            raise ValueError("No audio file provided.")
        
        if not audio_file.name.endswith(('.mp3', '.wav')):
            raise ValueError("Only .mp3 and .wav files are supported.")
        
        # Check file size (limit to 100MB)
        if audio_file.size > 100 * 1024 * 1024:
            raise ValueError("Audio file size exceeds 100MB limit.")
        return True
        
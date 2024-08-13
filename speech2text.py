import torch
from transformers import pipeline
import gradio as gr

# Initialize the speech-to-text pipeline from Hugging Face Transformers
# This uses the "openai/whisper-tiny.en" model for automatic speech recognition (ASR)
# The `chunk_length_s` parameter specifies the chunk length in seconds for processing

def transcript_audio(audio_file):
  pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
  )

  # Define the path to the audio file that needs to be transcribed
  sample = 'harvard.mp3'

  # Perform speech recognition on the audio file
  # The `batch_size=8` parameter indicates how many chunks are processed at a time
  # The result is stored in `prediction` with the key "text" containing the transcribed text
  prediction = pipe(sample, batch_size=8)["text"]

  # Print the transcribed text to the console
  return prediction

# Set up Gradio interface
audio_input = gr.Audio(sources="upload", type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output
# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=transcript_audio, 
                     inputs=audio_input, outputs=output_text, 
                     title="Audio Transcription App",
                     description="Upload the audio file")
# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=7860,share='True')


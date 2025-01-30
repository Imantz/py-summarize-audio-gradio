from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import gradio as gr

asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

summarization_pipe = pipeline("summarization", model="facebook/bart-large-cnn")

# Wrap Hugging Face models with LangChain
asr_model = HuggingFacePipeline(pipeline=asr_pipe)
summarizer_model = HuggingFacePipeline(pipeline=summarization_pipe)

# Define Summarization Prompt
summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following text in short bullet points. "
        "Keep it concise, highlighting only key points:\n\n"
        "{text}\n\n"
        "### Summary:\n- "
    )
)

# Create LLMChain for summarization
summarization_chain = LLMChain(llm=summarizer_model, prompt=summarization_prompt)

# Function to process audio
def transcript_and_summarize(audio_file):
    # Transcribe audio
    transcript = asr_pipe(audio_file)["text"]
    
    # Summarize text
    summary = summarization_chain.run(text=transcript)
    
    return transcript, summary

# Gradio UI
audio_input = gr.Audio(sources="upload", type="filepath")
output_transcript = gr.Textbox(label="Transcription")
output_summary = gr.Textbox(label="Summary")

iface = gr.Interface(
    fn=transcript_and_summarize,
    inputs=audio_input,
    outputs=[output_transcript, output_summary],
    title="STT + Summarization With LangChain",
    description="Upload an audio file to transcribe and summarize using LangChain."
)

iface.launch(server_name="0.0.0.0", server_port=7860)

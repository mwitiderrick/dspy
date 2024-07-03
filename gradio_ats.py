import gradio as gr
import google.generativeai as genai
import PyPDF2 
import os
import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field

llm = dspy.Google("models/gemini-pro", api_key="AIzaS\
")
dspy.settings.configure(lm=llm)

load_dotenv()

class Input(BaseModel):
    jd: str= Field(description="Job description")
    resume: str = Field(description="Resume containing the candidates skills and work experience")

class Output(BaseModel):
    jd_match: float = Field(ge=0, le=1,description="Percentage of the resume's match to the job description. Must be a number between 0 and 1")
    keywords: str = Field(description="Keywords missing in the candidates resume")
    profile_summary: str = Field(description="Summary of the candidate from provided resume. Mention missing keywords and how the candidate's profile can be improved")
    answer:str = Field(description="Whether the candidate should be hired. YES or NO")

class GenerateAnswer(dspy.Signature):
    """Evaluate the resume based on the given job description.
        Identify keywords that are missing from the candidate's resume based on the provided resume and job description. 
    """
    input: Input = dspy.InputField()
    output: Output = dspy.OutputField()

class ResumeReviewer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generateAnswer = dspy.TypedChainOfThought(GenerateAnswer)

    def forward(self, input):
        return self.generateAnswer(input=input)
        
resumeReviewer = ResumeReviewer()

def input_pdf_text(pdf):
    reader=PyPDF2.PdfReader(pdf)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

def ats_reviewer(jd, pdf):
    if pdf is not None:
        generateAnswer = dspy.Predict(GenerateAnswer)
        resume = input_pdf_text(pdf)
        llm_input = Input(jd=jd, resume=resume)
        response = resumeReviewer(llm_input)
        hire =  response.output.answer,
        jd_match =  response.output.jd_match,
        missing_keywords = response.output.keywords,
        profile_summary = response.output.profile_summary
        return hire,jd_match,missing_keywords,profile_summary
            
    return "Please upload a resume."


with gr.Blocks() as demo:    
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ## ATS Resume Reviewer
            """)
            jd = gr.Textbox(lines=10, label="Paste the Job Description")
            pdf = gr.File(label="Upload Your Resume (PDF)")
            btn = gr.Button("Submit")
            
        with gr.Column():
            gr.Markdown("""
            ### ATS Results
            """)
            hire = gr.Textbox(label="Hire")
            jd_match =  gr.Textbox(label="JD Match")
            missing_keywords = gr.Textbox(label="Missing Keywords")
            profile_summary = gr.Textbox(label="Profile Summary")
            
    btn.click(
        ats_reviewer,
        inputs=[jd,pdf],
        outputs=[hire,jd_match,missing_keywords,profile_summary],
    )


if __name__ == "__main__":
    demo.launch()



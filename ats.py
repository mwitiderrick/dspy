import streamlit as st
import PyPDF2 as pdf
import dspy
import requests
from pydantic import BaseModel, Field

llm = dspy.Google("models/gemini-pro", api_key="API_KEY")
dspy.settings.configure(lm=llm)

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

def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

## streamlit app
st.title("ATS Resume Reviewer")
st.text("Get Your Resume ATS Score")
jd=st.text_area("Paste the Job Description")
uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please upload the pdf")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        generateAnswer = dspy.Predict(GenerateAnswer)
        resume=input_pdf_text(uploaded_file)
        llm_input = Input(jd=jd,resume=resume,)
        response = resumeReviewer(llm_input)
        st.subheader("Response")
        st.text("Hire")
        st.text(response.output.answer)
        st.text("JD Match")
        st.text(response.output.jd_match)
        st.text("Missing Keywords")
        st.text(response.output.keywords)
        st.text("Profile Summary")
        st.write(response.output.profile_summary)

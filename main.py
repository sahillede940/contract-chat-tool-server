from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import json
from pydantic import BaseModel
from typing import List

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sanitize_json(response_string):
    response_string = (
        response_string.replace("```", "")
        .replace("json", "")
        .replace("Output Format:", "")
        .strip()
    )
    parsed_data = json.loads(response_string)
    return parsed_data


def read_pdf(file) -> str:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to generate questions using LangChain and OpenAI
def generate_questions_from_contract(pdf_text: str) -> str:
    """Sends a request to the LangChain API to generate questions from the PDF content."""
    # Initialize LangChain's OpenAI model interface
    llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key, model="gpt-4o")

    # Define the prompt for generating questions
    system_prompt = """
    You are analyzing a contract for a logistics or delivery company. Read the following contract and generate 100 relevant questions to clarify important terms and conditions. Also, provide potential answers based on the contract's content. The output must be in JSON format, where the questions and answers are represented as an array of objects. Each object should have two fields: "question" and "answer". Here's an example of the expected output format:

    1. Please ensure that the output follows this strict JSON format.
    Output Format:
    {
        "questions": [
            {
                "question": "",
                "answer": ""
            }
        ]
    }

    PDF Text: 
    """
    try:
        messages = [{"role": "system", "content": system_prompt + pdf_text}]
        response = llm.invoke(messages)
        return sanitize_json(response.content)
    except Exception as e:
        return f"Error: {str(e)}"


@app.post("/generate-questions/")
async def generate_questions(file: UploadFile = File(...)):
    """API endpoint that accepts a PDF, extracts the text, and generates questions."""
    try:
        pdf_text = read_pdf(file.file)
        questions = generate_questions_from_contract(pdf_text)

        return {"questions": questions.get("questions", []), "pdf_text": pdf_text}
    except Exception as e:
        return {"error": f"Failed to process the PDF: {str(e)}"}


class ChatContractRequest(BaseModel):
    messages: List[dict]
    pdf_text: str


@app.post("/chat-contract/")
async def chat_contract(request: ChatContractRequest):
    """API endpoint that accepts a list of messages and generates responses."""

    system_prompt = """
        You are a helpful assistant tasked with answering questions related to a contract. The information below has been extracted from a logistics or delivery contract. Use this data to answer any questions that a user might have. If the answer is not explicitly stated in the data, politely inform the user that the information is not available.
        Contract Information:
    """

    try:

        messages = request.messages
        pdf_text = request.pdf_text

        llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key, model="gpt-4o")
        messages.insert(0, {"role": "system", "content": system_prompt + pdf_text})
        response = llm.invoke(messages)
        return {
            "message": response.content,
        }
    except Exception as e:
        return f"Error: {str(e)}"

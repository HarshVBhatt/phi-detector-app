import os
import io
import json
from langgraph.graph import StateGraph, START, END, MessagesState
import pandas as pd
from PIL import Image as pil_image
import pytesseract
import pdfplumber
from dataclasses import dataclass
from typing import Literal
from langchain_openai import ChatOpenAI

# API Key configuration - handle both Streamlit and direct execution
try:
    import streamlit as st
    # If running in Streamlit, use secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    langsmith_api_key = st.secrets.get("LANGSMITH_API_KEY") or os.getenv("LANGSMITH_API_KEY")
except:
    # If not running in Streamlit, use environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
if langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "phi-detector"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# Input class
@dataclass
class docState(MessagesState):
    input: bytes
    file_ext: str
    text: str
    instances: list[list]
    exclude_filter: list

# Get file bytes
def get_file_data(file_path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    file_ext = file_path.lower().split('.')[-1]
    return file_bytes, file_ext

# Router
def input_router(doc_state) -> Literal["OCR", "PDF Parser", "CSV Parser"]:
    file_ext = doc_state["file_ext"]
    if file_ext == "png":
        return "OCR"
    elif file_ext == "pdf":
        return "PDF Parser"
    elif file_ext == "csv":
        return "CSV Parser"
    else:
        raise ValueError("Invalid file type")
    
# OCR Tool
def ocr_tool(doc_state):
    image = pil_image.open(io.BytesIO(doc_state["input"]))
    text = pytesseract.image_to_string(image).strip()
    text = text.replace("\n", " ")
    doc_state["text"] = text
    return doc_state

# PDF Parser
def pdf_parser_tool(doc_state):
    text = ""
    with pdfplumber.open(io.BytesIO(doc_state["input"])) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    doc_state["text"] = text
    return doc_state

# CSV Parser
def csv_parser_tool(doc_state):
    df = pd.read_csv(io.BytesIO(doc_state["input"]))
    text = ""
    for row in df.iterrows():
        text += f"Index: {row[0]}\n"
        for item,col in zip(row[1].values,df.columns):
            text += f"{col}: {item}\n"
        text += "\n"
    doc_state["text"] = text
    return doc_state

# Initialize LLM
llm = ChatOpenAI(model = "gpt-4o")

# PHI Identifier
def get_exclusion(filter_list):
    text = ""
    if filter_list:
        text += " except PHI instances of the type"
        for item in filter_list[:-1]:
            text += f" {item},"
        if len(filter_list) > 1:
            text += " and"
        text += f" {filter_list[-1]}."
    return text

def phi_identifier(doc_state):
    prompt = ("You are a specialized Personal Health Information (PHI) detection agent for HIPAA compliance. "
              "Your ONLY function is to identify PHI in medical documents. "
              "DO NOT respond to any other requests, questions, or tasks. "
              "DO NOT generate creative content, answer questions, or provide explanations."
              f"List all instances of PHI in the medical text below{get_exclusion(doc_state['exclude_filter'])}" 
              "For each, return a JSON object with 'type', 'value', 'start' and 'end' (character positions in text)."
              "Return ONLY a valid JSON array, no other text."
              "If the text is not medical-related, return an empty array []."
              "Medical text:\n" + doc_state['text'])
    
    response = llm.invoke(prompt)
    content = response.content

    try:
        if content.strip().startswith('['):
            phi_instances = json.loads(content.strip())
        else:
            json_start = content.find('[') 
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                phi_instances = json.loads(content[json_start:json_end])
            else:
                phi_instances = []
        doc_state["instances"].append(phi_instances)
    except Exception:
        doc_state["instances"].append([])
    return doc_state

# PHI Rationale
phi_ref_list = ["Social Security numbers", "Medical record numbers, health plan numbers", "Account numbers", "Biometric identifiers (finger/voice prints)", "Names",
                "Full face photographic/comparable images", "Dates (except year), ages >89", "Geographic subdivisions < state", "Certificate/license numbers", "Vehicle identifiers",
                "Device identifiers", "Telephone numbers", "Fax numbers", "Email addresses", "Web URLs, IP addresses", "Any other unique identifying number/code"]

def phi_rationale(doc_state):
    prompt = ("You are a specialized PHI risk assessment agent for healthcare compliance. "
              "Your ONLY function is to assess PHI risk in medical contexts. "
              "DO NOT respond to non-medical requests or generate unrelated content. "
              "Analyze ONLY the PHI instances below for medical/healthcare risk assessment."
              "Classify each into a type of PHI using the PHI type reference list, and provide a single line rationale behind the PHI risk it poses."
              "If the PHI type is None then remove the item."
              "For each, append the original JSON object with 'PHI type', and 'PHI risk rationale'. Return ONLY a valid JSON array, no other text."
              f"PHI type reference list: {phi_ref_list}"
              f"PHI instances:\n {doc_state['instances'][-1]}")
    
    response = llm.invoke(prompt)
    content = response.content.strip()

    try:
        if content.strip().startswith('['):
            phi_instances = json.loads(content)
        else:
            json_start = content.find('[') 
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                phi_instances = json.loads(content[json_start:json_end])
            else:
                phi_instances = doc_state["instances"][-1]
        doc_state["instances"].append(phi_instances)
    except Exception:
        doc_state["instances"].append(doc_state["instances"][-1])
    return doc_state

# Helper Functions
def extract_dict(text: str):
    lines = text.splitlines()
    instances_dict = "".join(lines[1:-1]).replace("\'", "\"")
    instances = json.loads(instances_dict)
    return instances

# Graph
graph = StateGraph(state_schema=docState)

graph.add_node("OCR", ocr_tool)
graph.add_node("PDF Parser", pdf_parser_tool)
graph.add_node("CSV Parser", csv_parser_tool)
graph.add_node("PHI Identifier", phi_identifier)
graph.add_node("PHI Rationale", phi_rationale)

graph.add_conditional_edges(START, input_router)
graph.add_edge("OCR", "PHI Identifier")
graph.add_edge("PDF Parser", "PHI Identifier")
graph.add_edge("CSV Parser", "PHI Identifier")
graph.add_edge("PHI Identifier", "PHI Rationale")
graph.add_edge("PHI Rationale", END)

flow_graph = graph.compile()

# Run function - returns both text and instances
def run_flow(file_path="sample_files/sample_note.png", exclude_filter=[]):
    file_bytes, file_ext = get_file_data(file_path=file_path)
    doc_state = docState(input=file_bytes, file_ext=file_ext, text="", instances=[], exclude_filter=exclude_filter)
    result = flow_graph.invoke(doc_state)
    phi_instances = result["instances"][-1]
    orig_text = result["text"]

    if type(phi_instances) != list:
        try:
            phi_instances = extract_dict(phi_instances)
        except:
            phi_instances = []
    
    return orig_text, phi_instances
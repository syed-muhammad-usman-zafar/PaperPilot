import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY, disable_streaming=True)

class ExtractionOutput(BaseModel):
    domain : list[str] = Field(description="The List of Domains Identified in the Prompt")
    research_methods: list[str] = Field(description="The List of Research Methods Identified in the Prompt")
    objectives: list[str] = Field(description="The List of Objectives Identified in the Prompt")
    data_types: list[str] = Field(description="The List of Sources from where I can extract information related to the topic discussed in the prompt")
    key_concepts: list[str] = Field(description="The List of Key Concepts Identified in the Prompt")
    method_type: str = Field(description="The Method of the Research Paper I will write. This can be empirical or theoretical or review or exploratory")
    objective_scope: str = Field(description="The Objective of the Research Paper I will write. This can be exploratory or confirmatory or analytical or comparative")


extraction_parser = JsonOutputParser(pydantic_object=ExtractionOutput)

prompt_template = PromptTemplate(
    input_variables=["prompt"],
    partial_variables={"format_instructions": extraction_parser.get_format_instructions()},
    template="""You are an expert research assistant designed to extract structured metadata from research-related prompts.

            Your task is to extract essential components from a research paper prompt. These components will help build a semantic understanding and a knowledge graph of the research idea.

            Return the result as a **valid JSON** with the following fields:

            - **domain**: List of academic or scientific domains (e.g., computer science, psychology, economics).
            - **research_methods**: List of research techniques or methodologies mentioned or implied (e.g., survey, experiment, case study, simulation).
            - **objectives**: Main goals or research questions described in the prompt.
            - **data_types**: Sources from where I can extract information related to the topic discussed in the prompt
            - **key_concepts**: Core concepts or keywords that define the theme of the research.
            - **method_type**: High-level categorization of research approach (must be one of: empirical, theoretical, review, exploratory).
            - **objective_scope**: The purpose or nature of the research objective (must be one of: exploratory, confirmatory, analytical, comparative).

            Use inference where necessary based on the prompt's content, even if explicit keywords are not used.

            Respond ONLY in the following JSON format: {format_instructions}

            **Prompt to Analyze**:{prompt}""")



def extract_with_llm(prompt):
    formatted_prompt = prompt_template.format(prompt=prompt)
    try:
        response = model.invoke(formatted_prompt)
        content = response.content if hasattr(response, "content") else response
        parsed = extraction_parser.parse(content)
        print(parsed)
        return parsed
    except Exception as e:
        print(f"[DEBUG] Gemini failed: {e}")
        return {} 
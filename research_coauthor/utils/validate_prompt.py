from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY, disable_streaming=True)

class ValidationOutput(BaseModel):
    write_request: bool = Field(..., description="True if the prompt asks to generate a research paper or provides a topic to write on.")

validation_parser = JsonOutputParser(pydantic_object=ValidationOutput)

validation_prompt_template = PromptTemplate(
    input_variables=["prompt"],
    partial_variables={"format_instructions": validation_parser.get_format_instructions()},
    template="You are a Research Paper Prompt Validator.\n\n"
        "Your task is to analyze a user prompt and determine whether it is suitable for writing a research paper.\n"
        "You should consider the following as valid prompts:\n"
        "- Direct requests to write or generate a research paper.\n"
        "- Topics or ideas clearly intended for a research paper (even if the word 'write' is not mentioned).\n\n"
        "Invalid prompts include:\n"
        "- General knowledge questions.\n"
        "- Requests that only seek explanations, definitions, or summaries.\n\n"
        "If the prompt is valid:\n"
        "- Set `write_request` to `true`.\n"
        "If the prompt is invalid:\n"
        "- Set `write_request` to `false`.\n"
        "**User Prompt:** {prompt}\n\n"
        "**Format Instructions: {format_instructions}")

def ValidatePrompt(prompt):
    validation_prompt = validation_prompt_template.format(prompt=prompt)
    response = model.invoke(validation_prompt)
    response = response.content if hasattr(response, "content") else response

    parsed_response = validation_parser.parse(response)
    return parsed_response
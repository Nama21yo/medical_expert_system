# app/subsymbolic/processor.py
import logging
from typing import List, Optional

from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from app.core.context_manager import get_conversation_chain
from langchain_core.output_parsers import PydanticOutputParser
import re

logger = logging.getLogger(__name__)

# --- Pydantic Models for Structured Output ---
class StructuredSymptom(BaseModel):
    symptom_name: str = Field(description="The MeTTa symbol for the symptom, e.g., 'ChestPain'.")
    strength: float = Field(description="The estimated strength of the symptom (0.0 to 1.0) based on the user's language.")
    confidence: float = Field(description="The estimated confidence (0.0 to 1.0) that the symptom is correctly identified.")

class SymptomAnalysisResult(BaseModel):
    extracted_symptoms: List[StructuredSymptom] = Field(description="A list of clearly identified symptoms.")
    ambiguous_terms: List[str] = Field(description="A list of vague terms that require clarification, e.g., 'pain', 'unwell'.")
    clarification_needed: bool = Field(description="True if more information is needed from the user.")

# --- Prompt Engineering Templates ---
EXTRACTION_PROMPT_TEMPLATE = """
You are an expert medical diagnosis assistant. Your role is to analyze a user's description of their condition and convert it into structured data for a symbolic AI reasoner.
The symbolic AI understands symptoms as MeTTa symbols (e.g., ChestPain, ShortnessOfBreath, RadiatingPain).

Analyze the user's latest input in the context of the entire conversation history.

Current Conversation:
{history}

User Input: {input}

Your task:
1.  Identify all distinct medical symptoms mentioned.
2.  For each symptom, map it to its corresponding MeTTa symbol.
3.  Estimate a 'strength' from 0.0 to 1.0 based on the user's wording (e.g., 'severe pain' is high strength, 'a little pain' is low).
4.  Estimate a 'confidence' from 0.0 to 1.0 in your identification of the symptom.
5.  If the user mentions a vague term like 'pain' without specifying its location or nature, or 'feeling unwell', add it to the 'ambiguous_terms' list. Do NOT add it to the extracted_symptoms list.
6.  Set 'clarification_needed' to true if you found any ambiguous terms.

{format_instructions}
"""

CLARIFICATION_PROMPT_TEMPLATE = """
You are a caring medical diagnosis assistant. Based on the conversation history, the user mentioned some vague symptoms.
Your goal is to ask a natural, empathetic follow-up question to get the specific details needed for diagnosis.

Conversation History:
{history}

Vague terms you need to clarify: {ambiguous_terms}

Based on this, generate a single, clear question to ask the user. For example, if they mentioned 'pain', ask where the pain is located and what it feels like.

Clarification Question:
"""

def extract_target_disease(user_input, known_diseases):
    """
    Returns the disease name if the user is asking about a specific disease, else None.
    """
    # Lowercase for matching
    text = user_input.lower()
    for disease in known_diseases:
        # Match "do i have X", "is it X", "could this be X", etc.
        patterns = [
            rf"do i have {disease.lower()}",
            rf"is (this|it) {disease.lower()}",
            rf"could (this|it) be {disease.lower()}",
            rf"am i experiencing {disease.lower()}",
        ]
        for pat in patterns:
            if re.search(pat, text):
                return disease
    return None

class SubsymbolicProcessor:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_chain = get_conversation_chain(session_id)
        self.parser = PydanticOutputParser(pydantic_object=SymptomAnalysisResult)

    def process_input(self, user_input: str) -> dict:
        """Processes user input, extracts symptoms, or generates a clarification question."""
        prompt = PromptTemplate(
            template=EXTRACTION_PROMPT_TEMPLATE,
            input_variables=["input", "history"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        # We need to format the prompt manually to use the Pydantic parser
        full_prompt = prompt.format(history=self.conversation_chain.memory.buffer_as_str, input=user_input)
        llm_output = self.conversation_chain.llm.invoke(full_prompt)
        
        try:
            parsed_output: SymptomAnalysisResult = self.parser.parse(llm_output.content)
            
            if parsed_output.clarification_needed:
                return self._ask_for_clarification(parsed_output.ambiguous_terms, user_input)
            else:
                # Add the successful interaction to memory
                self.conversation_chain.memory.save_context({"input": user_input}, {"output": "Symptoms successfully extracted."})
                return {"status": "success", "data": parsed_output.extracted_symptoms}

        except Exception as e:
            logger.error(f"Failed to parse LLM output: {e}\nOutput was: {llm_output.content}")
            return {"status": "error", "message": "I had trouble understanding that. Could you please rephrase?"}

    def _ask_for_clarification(self, ambiguous_terms: List[str], user_input: str) -> dict:
        """Generates a follow-up question for the user."""
        prompt = PromptTemplate.from_template(CLARIFICATION_PROMPT_TEMPLATE)
        
        chain = prompt | self.conversation_chain.llm
        
        clarification_question = chain.invoke({
            "history": self.conversation_chain.memory.buffer_as_str,
            "ambiguous_terms": ", ".join(ambiguous_terms)
        })

        # Add the clarification attempt to memory
        self.conversation_chain.memory.save_context({"input": user_input}, {"output": clarification_question.content})
        
        return {
            "status": "clarification_needed",
            "message": clarification_question.content
        }

    def curate_diagnosis_with_llm(self, extracted_symptoms, diagnosis_results):
        """
        Use the LLM to generate a user-friendly, empathetic summary of the diagnosis results.
        """
        # Prepare a prompt for the LLM
        prompt = PromptTemplate(
            template="""
You are a medical AI assistant. Given the following extracted symptoms and diagnostic hypotheses (with likelihood and confidence), write a single, confident, user-friendly paragraph summarizing the most likely conditions, their likelihood, and a safety disclaimer. Your answer should be direct and authoritative, based on the diagnostic hypotheses provided by the symbolic AI. Do not list all diseases, just the most relevant ones. Be concise and clear.

Extracted Symptoms: {symptoms}
Diagnostic Hypotheses: {diagnoses}

Summary:
""",
            input_variables=["symptoms", "diagnoses"]
        )

        # Format the diagnosis results for the LLM
        diagnoses_str = "\n".join(
            f"{d['disease']}: likelihood {d['strength']:.2f}, confidence {d['confidence']:.2f}"
            for d in diagnosis_results
        )
        symptoms_str = ", ".join(
            s["symptom_name"] if isinstance(s, dict) else getattr(s, "symptom_name", str(s))
            for s in extracted_symptoms
        )

        full_prompt = prompt.format(symptoms=symptoms_str, diagnoses=diagnoses_str)
        llm_output = self.conversation_chain.llm.invoke(full_prompt)
        return llm_output.content.strip()

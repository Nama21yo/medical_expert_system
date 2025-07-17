# app/main.py
import uuid
from fastapi import FastAPI, Request
from pydantic import BaseModel

from app.core.config import settings
from app.subsymbolic.processor import SubsymbolicProcessor
from app.symbolic.metta_integration import MedicalMeTTaEngine

app = FastAPI(title="Conversational Neurosymbolic Medical AI")
metta_engine = MedicalMeTTaEngine()

class DiagnoseRequest(BaseModel):
    text: str
    session_id: str

@app.post("/chat")
async def chat_endpoint(request: DiagnoseRequest):
    """
    Handles a turn in the conversation, either performing diagnosis or asking for clarification.
    """
    session_id = request.session_id
    user_input = request.text
    
    # 1. Initialize the subsymbolic processor for the current session
    processor = SubsymbolicProcessor(session_id=session_id)
    
    # 2. Process the user's input
    result = processor.process_input(user_input)
    
    # 3. Handle the result
    if result["status"] == "clarification_needed" or result["status"] == "error":
        # Return the question or error message to the user
        return {"response": result["message"], "type": "clarification"}
        
    elif result["status"] == "success":
        # We have clear symptoms, now run the symbolic reasoner
        structured_symptoms = result["data"]
        patient_id = f"Patient_{session_id}"

        # Reset state and add new symptoms
        metta_engine.reset_patient_state(patient_id)
        metta_engine.add_patient_symptoms(patient_id, structured_symptoms)

        # Run the full diagnosis
        diagnosis_results = metta_engine.run_diagnosis(patient_id)

        return {
            "response": diagnosis_results,
            "extracted_symptoms": [s.dict() for s in structured_symptoms],
            "type": "diagnosis"
        }

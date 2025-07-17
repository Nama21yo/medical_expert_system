import logging
import os
from hyperon import MeTTa, Environment, ExpressionAtom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalMeTTaEngine:
    def __init__(self, inference_path="pln_inference.metta"):
        logger.info("Initializing MeTTa Engine...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.metta = MeTTa(env_builder=Environment.custom_env(working_dir=current_dir)) # type: ignore
        
        kb_path = os.path.join(current_dir, "knowledge_base.metta")
        inference_path = os.path.join(current_dir, inference_path)
        self._load_metta_file(kb_path)
        self._load_metta_file(inference_path)
        logger.info("MeTTa Engine Initialized.")

    def _load_metta_file(self, file_path):
        try:
            logger.info(f"Loading MeTTa file: {file_path}")
            with open(file_path, "r") as f:
                content = f.read()
                result = self.metta.run(content)
                logger.info(f"Loaded {file_path} successfully. Result: {result}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise e

    def reset_patient_state(self, patient_id: str):
        logger.info(f"Resetting MeTTa engine state for patient: {patient_id}")
        clear_query = f"!(remove-atom &self (Evaluation (Predicate hasSymptom) (List {patient_id} $sym) $tv))"
        self.metta.run(clear_query)
        clear_query2 = f"!(remove-atom &self (Evaluation (Predicate possibleDisease) (List {patient_id} $disease) $tv))"
        self.metta.run(clear_query2)

    def add_patient_symptoms(self, patient_id: str, symptoms: list):
        logger.info(f"Adding structured symptoms for patient: {patient_id}")
        for symptom in symptoms:
            symptom_name = symptom.symptom_name if hasattr(symptom, 'symptom_name') else str(symptom)
            strength = symptom.strength if hasattr(symptom, 'strength') else 0.8
            confidence = symptom.confidence if hasattr(symptom, 'confidence') else 0.9
            metta_expr = f"!(add-atom &self (Evaluation (Predicate hasSymptom) (List {patient_id} {symptom_name}) (TV {strength} {confidence})))"
            result = self.metta.run(metta_expr)
            logger.info(f"Added: {symptom_name} for {patient_id}, result: {result}")

    def run_diagnosis(self, patient_id: str):
        logger.info(f"Starting diagnosis for {patient_id}...")
        symptoms_query = f"!(match &self (Evaluation (Predicate hasSymptom) (List {patient_id} $sym) $tv) $sym)"
        symptoms_result = self.metta.run(symptoms_query)
        logger.info(f"Found symptoms: {symptoms_result}")

        if symptoms_result and symptoms_result[0]:
            for symptom_result in symptoms_result[0]:
                symptom_name = str(symptom_result)
                logger.info(f"Processing symptom: {symptom_name}")
                fc_query = f"!(forward-chain {patient_id} {symptom_name})"
                fc_result = self.metta.run(fc_query)
                logger.info(f"Forward chaining result for {symptom_name}: {fc_result}")

        risk_adjustment_query = f"!(apply-risk-factors {patient_id})"
        self.metta.run(risk_adjustment_query)

        diagnosis_query = f"!(match &self (Evaluation (Predicate possibleDisease) (List {patient_id} $disease) $tv) (list $disease $tv))"
        results = self.metta.run(diagnosis_query)
        logger.info(f"MeTTa diagnosis query returned: {results}")

        diagnoses = []
        if results and results[0]:
            for result in results[0]:
                if isinstance(result, list) and len(result) >= 2:
                    disease = str(result[0])
                    tv = result[1]
                    strength = float(tv.get_children()[1]) if isinstance(tv, ExpressionAtom) else 0.5
                    confidence = float(tv.get_children()[2]) if isinstance(tv, ExpressionAtom) else 0.5
                    diagnoses.append({
                        "disease": disease,
                        "strength": f"{strength:.3f}",
                        "confidence": f"{confidence:.3f}"
                    })
                else:
                    logger.warning(f"Unexpected result format: {result}")

        diagnoses.sort(key=lambda x: float(x['strength']) * float(x['confidence']), reverse=True)
        logger.info(f"Diagnosis complete. Found: {diagnoses}")
        return diagnoses

    def explain_diagnosis(self, patient_id: str, disease: str):
        logger.info(f"Explaining diagnosis of {disease} for {patient_id}")
        explanation_query = f"!(get-diagnosis-explanation {patient_id} {disease})"
        explanation = self.metta.run(explanation_query)
        return explanation

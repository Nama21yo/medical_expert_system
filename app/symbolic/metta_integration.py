import logging
import os
import re
from hyperon import MeTTa, Environment, ExpressionAtom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalMeTTaEngine:
    def __init__(self, inference_path="pln_inference.metta"):
        logger.info("Initializing MeTTa Engine...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.metta = MeTTa(env_builder=Environment.custom_env(working_dir=current_dir))
        
        # Load knowledge base and inference engine
        kb_path = os.path.join(current_dir, "knowledge_base.metta")
        inference_path = os.path.join(current_dir, inference_path)
        
        self._load_metta_file(kb_path)
        self._load_metta_file(inference_path)
        logger.info("MeTTa Engine Initialized.")

    def _load_metta_file(self, file_path):
        """Load a MeTTa file and handle errors gracefully."""
        try:
            logger.info(f"Loading MeTTa file: {file_path}")
            with open(file_path, "r") as f:
                content = f.read()
                result = self.metta.run(content)
                logger.info(f"Loaded {file_path} successfully.")
                return result
        except FileNotFoundError:
            logger.error(f"MeTTa file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    def reset_patient_state(self, patient_id: str):
        """Reset patient state by removing existing symptoms and diagnoses."""
        logger.info(f"Resetting MeTTa engine state for patient: {patient_id}")
        
        # Remove existing symptoms
        clear_symptoms = f"!(remove-atom &self (Evaluation (Predicate hasSymptom) (List {patient_id} $sym) $tv))"
        self.metta.run(clear_symptoms)
        
        # Remove existing diagnoses
        clear_diagnoses = f"!(remove-atom &self (Evaluation (Predicate possibleDisease) (List {patient_id} $disease) $tv))"
        self.metta.run(clear_diagnoses)
        
        logger.info(f"Patient state reset for: {patient_id}")

    def add_patient_symptoms(self, patient_id: str, symptoms: list):
        """Add symptoms to the knowledge base for a patient."""
        logger.info(f"Adding symptoms for patient: {patient_id}")
        for symptom in symptoms:
            # Accepts either StructuredSymptom, dict, or string
            if hasattr(symptom, 'symptom_name'):
                symptom_name = symptom.symptom_name
                strength = getattr(symptom, 'strength', 0.8)
                confidence = getattr(symptom, 'confidence', 0.9)
            elif isinstance(symptom, dict):
                symptom_name = symptom.get('symptom_name')
                strength = symptom.get('strength', 0.8)
                confidence = symptom.get('confidence', 0.9)
            else:
                symptom_name = str(symptom)
                strength = 0.8
                confidence = 0.9

            metta_expr = f"!(add-atom &self (Evaluation (Predicate hasSymptom) (List {patient_id} {symptom_name}) (TV {strength} {confidence})))"
            self.metta.run(metta_expr)
            logger.info(f"Added symptom: {symptom_name} (strength: {strength}, confidence: {confidence})")

    def run_forward_diagnosis(self, patient_id: str):
        """Run forward chaining diagnosis using forward-chain-all."""
        logger.info(f"Running forward diagnosis for {patient_id}...")
        
        # Use forward-chain-all directly instead of diagnose-forward
        forward_query = f"!(forward-chain-all {patient_id})"
        results = self.metta.run(forward_query)
        logger.info(f"Forward chaining raw results: {results}")
        
        # Parse the results from forward-chain-all
        diagnoses = self._parse_forward_chain_results(results)
        
        # Also extract any diagnoses that were added to the knowledge base
        kb_diagnoses = self._extract_diagnoses_from_kb(patient_id)
        
        # Combine and deduplicate results
        all_diagnoses = self._combine_diagnoses(diagnoses, kb_diagnoses)
        
        logger.info(f"Forward chaining completed for {patient_id}, found {len(all_diagnoses)} diagnoses")
        return all_diagnoses

    def run_backward_diagnosis(self, patient_id: str, target_disease: str):
        """Run backward chaining diagnosis for a specific disease."""
        logger.info(f"Running backward diagnosis for {patient_id} targeting {target_disease}...")
        
        backward_query = f"!(backward-chain {patient_id} {target_disease})"
        logger.info(f"Backward chaining query: {backward_query}")
        result = self.metta.run(backward_query)
        logger.info(f"Backward chaining raw result: {result}")

        # If result is empty or only contains empty lists, return []
        if not result or (isinstance(result[0], list) and len(result[0]) == 0):
            logger.info(f"No evidence found for {target_disease} in backward chaining.")
            return []

        # If result is a list of TVs, parse all of them
        tv_list = result[0] if isinstance(result[0], list) else result
        diagnoses = []
        for tv in tv_list:
            strength, confidence = self._parse_tv_from_result(tv)
            diagnoses.append({
                "disease": target_disease,
                "strength": strength,
                "confidence": confidence,
                "score": strength * confidence
            })
        logger.info(f"Backward chaining completed: {target_disease} -> {diagnoses}")
        return diagnoses

    def run_diagnosis(self, patient_id: str, method='forward'):
        """Run diagnosis using specified method (forward or backward)."""
        logger.info(f"Starting {method} diagnosis for {patient_id}...")
        
        if method == 'forward':
            return self.run_forward_diagnosis(patient_id)
        elif method == 'backward':
            # For backward chaining, try multiple high-probability diseases
            target_diseases = ['MyocardialInfarction', 'Angina', 'PulmonaryEmbolism', 'GERD', 'PanicAttack']
            all_diagnoses = []
            for disease in target_diseases:
                diagnoses = self.run_backward_diagnosis(patient_id, disease)
                all_diagnoses.extend(diagnoses)
            
            # Sort by score
            all_diagnoses.sort(key=lambda x: x["score"], reverse=True)
            return all_diagnoses
        else:
            raise ValueError(f"Unknown diagnosis method: {method}")

    def _parse_forward_chain_results(self, results):
        """Parse the results from forward-chain-all which returns a list of (List disease TV) pairs."""
        diagnoses = []
        if not results or len(results) == 0:
            return diagnoses
        
        # Results should be a list of lists like: [(List MyocardialInfarction (TV 0.019 0.8844)), ...]
        result_list = results[0] if isinstance(results[0], list) else results
        
        for item in result_list:
            try:
                # Parse each item which should be (List disease (TV strength confidence))
                if isinstance(item, list) and len(item) >= 2:
                    disease = str(item[0])
                    tv_part = item[1]
                    
                    strength, confidence = self._parse_tv_from_result(tv_part)
                    
                    diagnoses.append({
                        "disease": disease,
                        "strength": strength,
                        "confidence": confidence,
                        "score": strength * confidence
                    })
                else:
                    # Try to parse as string representation
                    item_str = str(item)
                    disease, strength, confidence = self._parse_diagnosis_string(item_str)
                    if disease:
                        diagnoses.append({
                            "disease": disease,
                            "strength": strength,
                            "confidence": confidence,
                            "score": strength * confidence
                        })
            except Exception as e:
                logger.warning(f"Failed to parse diagnosis item {item}: {e}")
                continue
        
        return diagnoses

    def _parse_tv_from_result(self, tv_result):
        """Parse TV (Truth Value) from MeTTa result."""
        try:
            # If it's an ExpressionAtom, try to get children
            if hasattr(tv_result, 'get_children'):
                children = tv_result.get_children()
                if len(children) >= 3:  # TV strength confidence
                    strength = float(str(children[1]))
                    confidence = float(str(children[2]))
                    return strength, confidence
            
            # Fallback: parse as string
            tv_str = str(tv_result)
            numbers = re.findall(r'[\d.]+', tv_str)
            if len(numbers) >= 2:
                strength = float(numbers[0])
                confidence = float(numbers[1])
                return strength, confidence
            
        except Exception as e:
            logger.warning(f"Failed to parse TV {tv_result}: {e}")
        
        # Default values
        return 0.5, 0.5

    def _parse_diagnosis_string(self, diagnosis_str):
        """Parse diagnosis from string representation like '(List MyocardialInfarction (TV 0.019 0.8844))'."""
        try:
            # Extract disease name
            disease_match = re.search(r'List\s+(\w+)', diagnosis_str)
            disease = disease_match.group(1) if disease_match else None
            
            # Extract TV values
            tv_match = re.search(r'TV\s+([\d.]+)\s+([\d.]+)', diagnosis_str)
            if tv_match:
                strength = float(tv_match.group(1))
                confidence = float(tv_match.group(2))
                return disease, strength, confidence
            
        except Exception as e:
            logger.warning(f"Failed to parse diagnosis string {diagnosis_str}: {e}")
        
        return None, 0.5, 0.5

    def _extract_diagnoses_from_kb(self, patient_id: str):
        """Extract diagnoses from the knowledge base."""
        diagnosis_query = f"!(match &self (Evaluation (Predicate possibleDisease) (List {patient_id} $disease) $tv) (List $disease $tv))"
        results = self.metta.run(diagnosis_query)
        
        diagnoses = []
        if results and len(results) > 0:
            result_list = results[0] if isinstance(results[0], list) else results
            for item in result_list:
                if isinstance(item, list) and len(item) == 2:
                    disease = str(item[0])
                    tv = item[1]
                    strength, confidence = self._parse_tv_from_result(tv)
                    diagnoses.append({
                        "disease": disease,
                        "strength": strength,
                        "confidence": confidence,
                        "score": strength * confidence
                    })
        
        return diagnoses

    def _combine_diagnoses(self, diagnoses1, diagnoses2):
        """Combine and deduplicate diagnoses from multiple sources."""
        combined = {}
        
        # Add diagnoses from first source
        for diag in diagnoses1:
            disease = diag["disease"]
            combined[disease] = diag
        
        # Add or update diagnoses from second source
        for diag in diagnoses2:
            disease = diag["disease"]
            if disease in combined:
                # Take the one with higher confidence
                if diag["confidence"] > combined[disease]["confidence"]:
                    combined[disease] = diag
            else:
                combined[disease] = diag
        
        # Convert back to list and sort by score
        result = list(combined.values())
        result.sort(key=lambda x: x["score"], reverse=True)
        return result

    def curate_diagnosis_response(self, diagnoses, top_n=5):
        """Create a user-friendly answer from symbolic output."""
        if not diagnoses:
            return "Based on your symptoms, I could not identify a likely diagnosis. Please provide more details or consult a healthcare professional."

        # Filter out very low probability diagnoses
        filtered_diagnoses = [d for d in diagnoses if d["score"] > 0.01]
        
        if not filtered_diagnoses:
            return "The symptoms you described could indicate several conditions, but none with high confidence. Please consult a healthcare professional for proper evaluation."

        response = "Based on your symptoms, here are the most likely conditions:\n\n"
        
        for i, diag in enumerate(filtered_diagnoses[:top_n], 1):
            disease_name = self._format_disease_name(diag["disease"])
            probability = diag["score"]
            confidence_level = self._get_confidence_level(probability)
            
            response += f"{i}. **{disease_name}** - {confidence_level} likelihood ({probability:.1%})\n"
        
        response += "\n⚠️ **Important**: These are preliminary assessments based on symptom patterns. "
        response += "Please consult a healthcare professional for proper diagnosis and treatment.\n\n"
        
        # Add some context based on top diagnosis
        if filtered_diagnoses:
            top_diagnosis = filtered_diagnoses[0]
            response += self._add_diagnosis_context(top_diagnosis["disease"])
        
        return response

    def _format_disease_name(self, disease):
        """Format disease name for display."""
        # Convert CamelCase to readable format
        formatted = re.sub(r'([A-Z])', r' \1', disease).strip()
        # Handle special cases
        formatted = formatted.replace('G E R D', 'GERD (Acid Reflux)')
        formatted = formatted.replace('Myocardial Infarction', 'Heart Attack')
        formatted = formatted.replace('Pulmonary Embolism', 'Blood Clot in Lung')
        return formatted

    def _get_confidence_level(self, probability):
        """Convert probability to confidence level."""
        if probability > 0.7:
            return "Very High"
        elif probability > 0.4:
            return "High"
        elif probability > 0.2:
            return "Moderate"
        elif probability > 0.1:
            return "Low"
        else:
            return "Very Low"

    def _add_diagnosis_context(self, disease):
        """Add contextual information for the top diagnosis."""
        context_map = {
            "MyocardialInfarction": "If you're experiencing severe chest pain, especially with shortness of breath, seek immediate medical attention.",
            "Angina": "Chest pain related to reduced blood flow to the heart. Consider lifestyle changes and medical evaluation.",
            "PulmonaryEmbolism": "A serious condition requiring immediate medical attention if suspected.",
            "GERD": "Acid reflux can cause chest pain. Consider dietary modifications and over-the-counter treatments.",
            "PanicAttack": "Anxiety-related chest pain. Breathing exercises and stress management may help.",
            "Asthma": "Respiratory condition that may require inhaler medication and trigger avoidance."
        }
        
        return context_map.get(disease, "Consider consulting a healthcare professional for proper evaluation and treatment.")

    def explain_diagnosis(self, patient_id: str, disease: str):
        """Get explanation for a specific diagnosis."""
        logger.info(f"Explaining diagnosis of {disease} for {patient_id}")
        explanation_query = f"!(explain-diagnosis {patient_id} {disease})"
        result = self.metta.run(explanation_query)
        return self._parse_explanation(result)

    def _parse_explanation(self, result):
        """Parse explanation results from MeTTa."""
        if not result:
            return {"explanation": "No explanation available"}
        try:
            return {
                "explanation": str(result),
                "supporting_symptoms": [],
                "risk_factors": [],
                "prevalence": "Unknown"
            }
        except Exception as e:
            logger.warning(f"Failed to parse explanation: {e}")
            return {"explanation": "Error parsing explanation"}

    def get_patient_symptoms(self, patient_id: str):
        """Get all symptoms for a patient."""
        symptoms_query = f"!(match &self (Evaluation (Predicate hasSymptom) (List {patient_id} $symptom) $tv) (List $symptom $tv))"
        result = self.metta.run(symptoms_query)
        symptoms = []
        if result and len(result) > 0:
            result_list = result[0] if isinstance(result, list) else result
            for item in result_list:
                if isinstance(item, list) and len(item) >= 2:
                    symptom_name = str(item[0])
                    symptoms.append(symptom_name)
        return symptoms

    def debug_knowledge_base(self):
        """Debug function to inspect the knowledge base."""
        logger.info("Debugging knowledge base...")
        all_atoms_query = "!(get-atoms &self)"
        result = self.metta.run(all_atoms_query)
        logger.info(f"Knowledge base contains {len(result) if result else 0} atoms")
        return result

# Example usage function
def diagnose_user_symptoms(user_id, extracted_symptoms):
    """
    Given a user_id and a list of extracted symptoms,
    reset the symbolic engine, add symptoms, run diagnosis, and return a curated answer.
    
    Example:
        symptoms = ["ChestPain", "ShortnessOfBreath"]
        response = diagnose_user_symptoms("user123", symptoms)
    """
    try:
        engine = MedicalMeTTaEngine()
        engine.reset_patient_state(user_id)
        engine.add_patient_symptoms(user_id, extracted_symptoms)
        
        # Run forward chaining diagnosis
        diagnoses = engine.run_forward_diagnosis(user_id)
        
        # Return curated response
        return engine.curate_diagnosis_response(diagnoses)
        
    except Exception as e:
        logger.error(f"Error in diagnosis: {e}")
        return "I apologize, but I encountered an error while processing your symptoms. Please try again or consult a healthcare professional."

# Example usage for testing
if __name__ == "__main__":
    # Test with the example symptoms
    test_symptoms = ["ChestPain", "ShortnessOfBreath", "Sweating"]
    response = diagnose_user_symptoms("TestUser", test_symptoms)
    print("Diagnosis Response:")
    print(response)

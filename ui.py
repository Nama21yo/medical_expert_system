
import gradio as gr
import requests
import uuid

API_URL = "http://127.0.0.1:8000/chat"

def format_diagnosis(data):
    """Formats the JSON diagnosis into readable markdown."""
    if not data["response"]:
        return "No specific diagnosis could be inferred based on the current information."
    
    output = "### Diagnostic Hypotheses\n"
    output += f"**Extracted Symptoms:** `{[s['symptom_name'] for s in data['extracted_symptoms']]}`\n\n---\n\n"
    
    for diag in data["response"]:
        output += (f"- **Disease:** `{diag['disease']}`\n"
                   f"  - **Likelihood (Strength):** {float(diag['strength']):.2f}\n"
                   f"  - **Confidence:** {float(diag['confidence']):.2f}\n")
    return output

def chat_function(message, history, session_id):
    """The core function for the Gradio ChatInterface."""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    payload = {"text": message, "session_id": session_id}
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if data["type"] == "diagnosis":
            bot_message = format_diagnosis(data)
        else: # clarification or error
            bot_message = data["response"]
            
        return bot_message, session_id
        
    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to backend. Is it running? Details: {e}", session_id

def main_interface():
    with gr.Blocks(theme=gr.themes.Soft(), css="#chatbot { min-height: 600px; }") as demo:
        gr.Markdown("# Conversational Medical Diagnosis AI")
        gr.Markdown("Start by describing your symptoms. The AI may ask follow-up questions for clarity.")
        
        session_id_state = gr.State(lambda: str(uuid.uuid4()))
        chatbot = gr.Chatbot(elem_id="chatbot")
        msg = gr.Textbox(label="Your Message", placeholder="e.g., I have a sharp pain in my chest and I feel sick.")

        def respond(message, chat_history, session_id):
            bot_message, new_session_id = chat_function(message, chat_history, session_id)
            chat_history.append((message, bot_message))
            return "", chat_history, new_session_id

        msg.submit(respond, [msg, chatbot, session_id_state], [msg, chatbot, session_id_state])

    demo.launch()

if __name__ == "__main__":
    main_interface()

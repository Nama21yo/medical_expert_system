# Medical Expert System

## Overview
A neuro-symbolic AI system integrating GeminiLLM/ChromaDB (sub-symbolic) and MeTTa/PLN (symbolic) for medical diagnosis.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Install spaCy model: `python -m spacy download en_core_web_sm`
3. Install MeTTa (see [MeTTa GitHub](https://github.com/trueagi-io/metta)).
4. Run API: `python src/api.py`
5. Launch UI: `python src/ui.py`

## Usage
Enter symptoms in the Gradio UI (e.g., "I have chest pain and shortness of breath") to receive a diagnosis.

## Structure
- `data/`: Knowledge base and ChromaDB storage.
- `src/`: Core scripts (sub-symbolic, MeTTa interface, API, UI).
- `tests/`: Unit tests.
- `docs/`: Documentation.

## Debugging
MeTTa includes `trace` and `println` for debugging inference steps.
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from backend.rag import retrieve_context
from backend.config import IBM_API_KEY, IBM_PROJECT_ID, IBM_URL

credentials = Credentials(
    api_key=IBM_API_KEY,
    url=IBM_URL
)

model = ModelInference(
    model_id="ibm/granite-4-h-small",
    credentials=credentials,
    project_id=IBM_PROJECT_ID
)

def ask_granite(question, context, sources):

    retrieved = retrieve_context(question)

    context = retrieved["context"]
    sources = retrieved["sources"]

    prompt = f"""
You are CampusCompanion, an AI-powered college assistant.

Answer ONLY using the information provided below.

If the answer cannot be found in the context,
say:

"I couldn't find that information in the college handbook."

Context:
{context}

Student Question:
{question}

Answer:
"""

    response = model.generate_text(
        prompt=prompt,
        params={
            "max_new_tokens":300,
            "temperature":0.3
        }
    )
    return {
    "answer": response,
    "sources": sources
}
import os
from dotenv import load_dotenv
import pandas as pd
import google.genai as genai
from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel, Field
import json
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Any, Tuple

class SpamResponse(BaseModel):
    is_spam: bool = Field(description="Whether the message is spam.")
    explanation: str = Field(
        description="An explanation of why the message is or is not spam."
    )

def initialize_genai_client() -> genai.Client:
    """
    Loads the Gemini API key from a .env file and returns an initialized
    genai.Client object.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env file or environment variables."
        )
    return genai.Client(api_key=api_key)

def load_messages(file_path: str) -> List[Dict[str, Any]]:
    """Loads messages from a CSV file."""
    try:
        messages_df = pd.read_csv(file_path)
        messages_df["is_spam"] = messages_df["spam"].astype(bool)
        messages_df["image_path"] = messages_df["image_path"].fillna("")
        messages_df["expected_explanation"] = messages_df[
            "expected_explanation"
        ].fillna("")
        return messages_df[
            ["text", "is_spam", "image_path", "expected_explanation"]
        ].to_dict(orient="records")
    except FileNotFoundError:
        print(
            f"{file_path} not found. Please create it with 'text', 'spam', "
            f"'image_path', and 'expected_explanation' columns."
        )
        return []

def get_spam_response_from_ai(
    client: genai.Client,
    full_prompt: str,
    model_name: str,
    image_path: str = None,
) -> SpamResponse:
    """Calls the Gemini AI and returns a SpamResponse object."""
    contents = [Part.from_text(text=full_prompt)]
    if image_path:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_part = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        contents.append(image_part)
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=GenerateContentConfig(
            response_mime_type="application/json", response_schema=SpamResponse
        ),
    )
    response_json = json.loads(response.text)
    return SpamResponse(**response_json)

def calculate_similarity(
    gemini_explanation: str,
    expected_explanation: str,
    sbert_model: SentenceTransformer,
) -> float:
    """Calculates the cosine similarity between two explanations."""
    embeddings1 = sbert_model.encode(gemini_explanation, convert_to_tensor=True)
    embeddings2 = sbert_model.encode(
        expected_explanation, convert_to_tensor=True
    )
    cosine_similarity = util.cos_sim(embeddings1, embeddings2).item()
    return cosine_similarity

def evaluate_models(
    client: genai.Client,
    messages: List[Dict[str, Any]],
    model_names: List[str],
    sbert_model: SentenceTransformer,
    prompt_template: str,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, List[float]]]:
    """
    Loops through messages and models to perform evaluation.

    Returns:
        A tuple containing dictionaries for correct identifications,
        total evaluations, and explanation similarities.
    """
    correct_identifications = {model_name: 0 for model_name in model_names}
    total_evaluations = {model_name: 0 for model_name in model_names}
    explanation_similarities = {model_name: [] for model_name in model_names}
    for message_data in messages:
        message_text = message_data["text"]
        is_spam = message_data["is_spam"]
        image_path = message_data.get("image_path")
        full_prompt = prompt_template.format(message_text)
        for model_name in model_names:
            spam_response = get_spam_response_from_ai(
                client, full_prompt, model_name, image_path
            )
            if spam_response.is_spam == is_spam:
                correct_identifications[model_name] += 1
            cosine_similarity = calculate_similarity(
                spam_response.explanation,
                message_data["expected_explanation"],
                sbert_model,
            )
            explanation_similarities[model_name].append(cosine_similarity)
            total_evaluations[model_name] += 1
    return (
        correct_identifications,
        total_evaluations,
        explanation_similarities,
    )

def print_accuracy_results(
    model_names: List[str],
    correct_identifications: Dict[str, int],
    total_evaluations: Dict[str, int],
):
    """Prints the final spam identification accuracy results."""
    print("--- Spam Identification Accuracy ---")
    for model_name in model_names:
        total = total_evaluations[model_name]
        correct = correct_identifications[model_name]
        if total > 0:
            accuracy = (correct / total) * 100
            print(
                f'Model "{model_name}": ' 
                f"Correctly identified {correct}/{total} " 
                f"({accuracy:.0f}%)"
            )
        else:
            print(f'Model "{model_name}": No evaluations for this model.')

def print_explanation_similarity_results(
    model_names: List[str], explanation_similarities: Dict[str, List[float]]
):
    """Prints the average explanation similarity results."""
    print("\n--- Explanation Similarity (Cosine) ---")
    for model_name in model_names:
        if explanation_similarities[model_name]:
            avg_similarity = sum(explanation_similarities[model_name]) / len(
                explanation_similarities[model_name]
            )
            print(
                f'Model "{model_name}": Average Explanation Similarity: ' 
                f"{avg_similarity * 100:.0f}%"
            )
        else:
            print(
                f'Model "{model_name}": No explanation similarity data for ' 
                f"this model."
            )

def main():
    """Runs the main model evaluation script."""
    client = initialize_genai_client()
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    messages = load_messages("messages.csv")
    prompt_template = (
        "Analyze the text and image (if there is one) of this social media "
        "post. Determine if either the text or the image indicates that the "
        "post is commercial spam. Text: {}"
    )
    model_names = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    (
        correct_identifications,
        total_evaluations,
        explanation_similarities,
    ) = evaluate_models(
        client, messages, model_names, sbert_model, prompt_template
    )
    print_accuracy_results(
        model_names, correct_identifications, total_evaluations
    )
    print_explanation_similarity_results(
        model_names, explanation_similarities
    )

if __name__ == "__main__":
    main()

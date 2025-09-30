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
        expected_explanation,
        convert_to_tensor=True,
    )
    cosine_similarity = util.cos_sim(embeddings1, embeddings2).item()
    return cosine_similarity

def evaluate_prompts(
    client: genai.Client,
    messages: List[Dict[str, Any]],
    prompt_templates: List[str],
    sbert_model: SentenceTransformer,
    model_name: str,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, List[float]]]:
    """
    Loops through messages and prompts to perform evaluation.

    Returns:
        A tuple containing dictionaries for correct identifications,
        total evaluations, and explanation similarities.
    """
    correct_identifications = {template: 0 for template in prompt_templates}
    total_evaluations = {template: 0 for template in prompt_templates}
    explanation_similarities = {template: [] for template in prompt_templates}
    for message_data in messages:
        message_text = message_data["text"]
        is_spam = message_data["is_spam"]
        image_path = message_data.get("image_path")
        for prompt_template in prompt_templates:
            full_prompt = prompt_template.format(message_text)
            spam_response = get_spam_response_from_ai(
                client, full_prompt, model_name, image_path
            )
            if spam_response.is_spam == is_spam:
                correct_identifications[prompt_template] += 1
            cosine_similarity = calculate_similarity(
                spam_response.explanation,
                message_data["expected_explanation"],
                sbert_model,
            )
            explanation_similarities[prompt_template].append(cosine_similarity)
            total_evaluations[prompt_template] += 1    
    return (
        correct_identifications,
        total_evaluations,
        explanation_similarities,
    )

def print_accuracy_results(
    prompt_templates: List[str],
    correct_identifications: Dict[str, int],
    total_evaluations: Dict[str, int],
):
    """Prints the final spam identification accuracy results."""
    print("--- Spam Identification Accuracy ---")
    for template in prompt_templates:
        total = total_evaluations[template]
        correct = correct_identifications[template]
        if total > 0:
            accuracy = (correct / total) * 100
            print(
                f'Prompt "{template}": ' 
                f"Correctly identified {correct}/{total} " 
                f"({accuracy:.0f}%)"
            )
        else:
            print(f'Prompt "{template}": No evaluations for this prompt.')

def print_explanation_similarity_results(
    prompt_templates: List[str],
    explanation_similarities: Dict[str, List[float]],
):
    """Prints the average explanation similarity results."""
    print("\n--- Explanation Similarity (Cosine) ---")
    for template in prompt_templates:
        if explanation_similarities[template]:
            avg_similarity = sum(explanation_similarities[template]) / len(
                explanation_similarities[template]
            )
            print(
                f'Prompt "{template}": Average Explanation Similarity: ' 
                f"{avg_similarity * 100:.0f}%"
            )
        else:
            print(
                f'Prompt "{template}": No explanation similarity data for ' 
                f"this prompt."
            )

def main():
    """Runs the main prompt evaluation script."""
    client = initialize_genai_client()
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    messages = load_messages("messages.csv")
    prompt_templates = [
        "Is this a commercial spam message? {}",
        "Analyze this message: {}",
        (
            "Analyze the text and image (if there is one) of this social media "
            "post. Determine if either the text or the image indicates that the "
            "post is commercial spam. Text: {}"
        ),
    ]
    model_name = "gemini-2.5-flash"
    (
        correct_identifications, total_evaluations, explanation_similarities
    ) = evaluate_prompts(
        client, messages, prompt_templates, sbert_model, model_name
    )
    print_accuracy_results(
        prompt_templates, correct_identifications, total_evaluations
    )
    print_explanation_similarity_results(
        prompt_templates, explanation_similarities
    )

if __name__ == "__main__":
    main()

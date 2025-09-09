# AI Model and Prompt Evaluation Toolkit

This repository contains a set of Python scripts designed to demonstrate advanced techniques for evaluating generative AI models and prompts. It serves as the official code for an upcoming Google Cloud blog post and video tutorial.

The primary goal of this project is to provide a clear, code-based guide for developers on how to handle more complex evaluation scenarios that go beyond simple text-in, text-out interactions.

## Key Features

This toolkit demonstrates how to:

*   **Enforce Structured JSON Output:** Use Pydantic models to define a strict schema for the AI model's JSON output, ensuring reliability and ease of parsing.
*   **Handle Multimodal Inputs:** Send both text and images to a multimodal AI model for evaluation.
*   **Calculate Evaluation Metrics:**
    *   **Accuracy:** Measure the correctness of a model's classification (e.g., `is_spam`).
    *   **Semantic Similarity:** Use sentence transformers to calculate the cosine similarity between the model's generated text and a reference explanation.
*   **Perform Comparative Analysis:**
    *   `compare_prompts.py`: Evaluate multiple prompts against a single AI model.
    *   `compare_models.py`: Evaluate multiple AI models using a single prompt.

## Getting Started

Follow these steps to set up and run the evaluation scripts.

### 1. Prerequisites

*   Python 3.8+
*   An active Google Cloud project with the Vertex AI API enabled.
*   A Gemini API key.

### 2. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 3. Set Up the Environment

It is recommended to use a Python virtual environment to manage dependencies.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 4. Configure Your API Key

The scripts load your Gemini API key from a `.env` file.

1.  Create a new file named `.env` in the root of the project directory.
2.  Add your API key to the file as follows:

```
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

The `.gitignore` file is configured to prevent the `.env` file from being committed to the repository.

## Usage

This project includes two main scripts for performing comparative evaluations.

### Comparing Prompts

To evaluate how different prompts perform with a single model, run `compare_prompts.py`:

```bash
python compare_prompts.py
```

This script will loop through the `prompt_templates` defined in the file and print the accuracy and explanation similarity for each.

### Comparing Models

To evaluate how different models perform with a single prompt, run `compare_models.py`:

```bash
python compare_models.py
```

This script will loop through the `model_names` defined in the file and report the performance metrics for each model.

## Customization

This toolkit is designed to be a starting point for your own evaluation needs.

*   **Use Your Own Data:** Modify the `messages.csv` file to include your own text, images, and ground truth data.
*   **Change the Schema:** Update the `SpamResponse` Pydantic model in the scripts to match the JSON structure your use case requires.
*   **Adapt the Logic:** The core functions for calling the AI, calculating similarity, and printing results can be easily adapted for different evaluation tasks.

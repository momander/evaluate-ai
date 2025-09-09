# Product Requirements Document: AI Model and Prompt Evaluation App

## 1. Vision

This application will serve as a comprehensive guide for Google Cloud customers, demonstrating how to write code for evaluating their AI models and prompts. It will be the subject of a detailed blog post and a video tutorial. The application will specifically focus on advanced AI calling techniques, such as handling structured JSON output and processing image inputs, addressing a gap in existing tools that primarily cover simpler, text-based scenarios.

## 2. Problem

Existing online tools for evaluating AI models and prompts are limited to simple text-in, text-out scenarios. They lack the capability to handle more complex and increasingly common use cases, such as inputs that include images (multimodal inputs) and outputs that are structured in JSON format. This application will bridge that gap by providing developers with a clear, code-based example of how to manage these advanced evaluation workflows.

## 3. Goals

The primary goals of this application are to:

*   Demonstrate how to effectively parse and validate structured JSON output received from an AI model.
*   Illustrate how to use images as inputs for AI models, moving beyond simple text-based prompts.
*   Provide a clear and reusable code base that developers can adapt for their own evaluation needs.

## 4. Features

The application demonstrates the following core features through a spam detection use case:

*   **Structured JSON Output:**
    *   Uses a Pydantic model (`SpamResponse`) to define a required JSON schema.
    *   Configures the AI model to return a JSON object containing a boolean (`is_spam`) and a string (`explanation`).
*   **Multimodal Input Handling:**
    *   Processes inputs that contain both text and an image, sending them to a multimodal AI model.
*   **Evaluation Metrics:**
    *   **Accuracy:** Measures the correctness of the boolean `is_spam` field against a ground truth value.
    *   **Semantic Similarity:** Calculates the cosine similarity between the model's generated `explanation` and a reference explanation to measure how closely their meanings align.
*   **Comparative Analysis Scripts:**
    *   `compare_prompts.py`: A script to evaluate the performance of multiple different prompts against a single model.
    *   `compare_models.py`: A script to evaluate the performance of multiple different models using a single prompt.

## 5. User Journey

The envisioned workflow for a developer using this application is as follows:

1.  **Acquire the Code:** Clone the application's repository from its source.
2.  **Setup Environment:** Install all necessary dependencies using a provided `requirements.txt` file.
3.  **Run a Demo:** Execute a command-line script to run a pre-configured evaluation example and see the output.
4.  **Customize:** Modify the code to suit their specific evaluation needs and AI model.
5.  **Integrate Data:** Add their own test data for evaluation.

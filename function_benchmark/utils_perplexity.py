import re
import json
import requests
import re
import json
import requests


def extract_model_names(json_path, output_model_names_path):
    """Extract model names from a JSON file and save them to another JSON file."""
    try:
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        model_names = [
            {"model_name": entry.get("model_name")}
            for entry in json_data
            if entry.get("model_name")
        ]

        with open(output_model_names_path, 'w') as output_file:
            json.dump(model_names, output_file, indent=4)

        print(f"Model names extracted and saved to {output_model_names_path}")

    except Exception as e:
        print(f"An error occurred while extracting model names: {e}")


def clean_json_response(json_str):
    """Clean numerical fields in text JSON responses."""
    # Replace "B" with "e9" for billions
    json_str = re.sub(
        r'(\d+(\.\d+)?)\s*[Bb]', lambda m: str(float(m.group(1)) * 1e9), json_str
    )
    # Replace "K" with "e3" for thousands
    json_str = re.sub(
        r'(\d+(\.\d+)?)\s*[Kk]', lambda m: str(float(m.group(1)) * 1e3), json_str
    )
    # Replace "None" (text) with "null" (valid JSON value)
    json_str = json_str.replace("None", "null")
    return json_str


def extract_json_from_response(response_text):
    """Extract JSON content between ```json``` markers in API responses."""
    json_match = re.search(
        r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL
    )
    if json_match:
        return json_match.group(1)
    return None


def get_prompt(model_name):
    """Generate the prompt for the API call."""
    return f"""
Please return information about the model '{model_name}' in strict JSON format, without extra explanation, using the following format:

{{
    "model_name": "{model_name}",
    "model_type": "Type of model (text, image, video, multimodal)",
    "nbr_parameters_active": "Number of active parameters (numeric only)",
    "nbr_parameters_total": "Total number of parameters (numeric only)",
    "context_window": "Maximum context length in tokens (numeric only)",
    "date_release": "Release date in the format YYYY-MM-DD"
}}
Very important rules to follow:
- The context window must be specified as an integer, so for example 32k becomes 32000.
- The number of parameters must be specified as an integer, so 32B becomes 32000000000, for example.
- In your searches, use the results of the most recent online searches. The most reliable sources are the companies that develop the models.
- If you don't know an answer, write null. Be careful not to generate hallucinations.
- Don't write any comments in the JSON.
"""


def call_perplexity_api(model_name, api_key):
    """Call the Perplexity API for a single model name."""
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an API expert in data extraction and generating "
                    "model details in strict JSON format."
                ),
            },
            {"role": "user", "content": get_prompt(model_name)},
        ],
        "max_tokens": 150,
        "temperature": 0.4,
        "top_p": 0.9,
        "return_citations": False,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers)
    return response


def call_perplexity_api_and_update(
    model_names_path, output_updated_models_path, api_key
):
    """Call the Perplexity API and update models with new information."""
    try:
        with open(model_names_path, 'r') as file:
            model_names_data = json.load(file)

        updated_models = []

        for model_entry in model_names_data:
            model_name = model_entry.get("model_name")

            try:
                response = call_perplexity_api(model_name, api_key)

                if response.status_code == 200:
                    result = response.json()
                    response_text = result['choices'][0]['message']['content']

                    # Display the raw API response for diagnostic purposes
                    print(f"Raw API response for model '{model_name}': {response_text}")

                    # Extract the JSON between the ```json``` markers
                    extracted_json_str = extract_json_from_response(response_text)

                    if extracted_json_str:
                        # Clean and parse the extracted JSON content
                        cleaned_response = clean_json_response(extracted_json_str)
                        extracted_json = json.loads(cleaned_response)

                        # Add the extracted information to the model
                        updated_model_entry = {
                            "model_name": model_name,
                            "model_type": extracted_json.get("model_type"),
                            "nbr_parameters_active": extracted_json.get("nbr_parameters_active"),
                            "nbr_parameters_total": extracted_json.get("nbr_parameters_total"),
                            "context_window": extracted_json.get("context_window"),
                            "date_release": extracted_json.get("date_release"),
                        }
                        updated_models.append(updated_model_entry)
                    else:
                        print(f"No valid JSON data found for model '{model_name}'")
                else:
                    print(
                        f"Failed to fetch data for model '{model_name}', "
                        f"status code: {response.status_code}"
                    )

            except Exception as e:
                print(f"An error occurred while processing model '{model_name}': {e}")
                continue  # Move to the next model

        # Save the updated results to a JSON file
        with open(output_updated_models_path, 'w') as output_file:
            json.dump(updated_models, output_file, indent=4)

        print(f"Updated model information saved to {output_updated_models_path}")

    except Exception as e:
        print(f"An error occurred while processing the model list: {e}")


def process_model_information(
    original_json_path, model_names_output_path, updated_models_output_path, api_key
):
    """Main function to run the extraction and update process."""
    # Step 1: Extract model names and save them to a JSON file
    extract_model_names(original_json_path, model_names_output_path)

    # Step 2: Call the Perplexity API and update model information
    call_perplexity_api_and_update(
        model_names_output_path, updated_models_output_path, api_key
    )




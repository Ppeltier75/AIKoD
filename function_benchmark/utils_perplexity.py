import re
import csv
import json
import requests


def process_id_name(id_name):
    """Process id_name by removing 'unknown' and replacing '-' with spaces, and removing non-essential segments."""
    # Split the id_name by '-'
    parts = id_name.split('-')
    # Remove 'unknown' parts and non-essential segments like 'false', 'true'
    parts = [part for part in parts if part.lower() not in ['unknown', 'false', 'true']]
    # Replace '-' with spaces
    processed_name = ' '.join(parts)
    return processed_name


def extract_model_info_from_csv(csv_path, output_model_info_path):
    """
    Extract model names and company names from a CSV file,
    process the id_name, and save them to a JSON file.
    """
    try:
        model_info_list = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                id_name = row.get('id_name', '')
                company = row.get('company', '')
                if id_name and company:
                    processed_name = process_id_name(id_name)
                    model_info_list.append({
                        "model_name": processed_name.strip(),
                        "company": company.strip()
                    })
        with open(output_model_info_path, 'w', encoding='utf-8') as output_file:
            json.dump(model_info_list, output_file, indent=4)
        print(f"Model info extracted and saved to {output_model_info_path}")
    except Exception as e:
        print(f"An error occurred while extracting model info: {e}")


def clean_json_response(json_str):
    """Clean JSON responses to ensure proper boolean values."""
    # Replace 'None' with 'null'
    json_str = json_str.replace('None', 'null')
    return json_str


def extract_json_from_response(response_text):
    """Extract JSON content between ```json``` or ``` markers in API responses."""
    # Regex to match both ```json and ``` for JSON blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if json_match:
        return json_match.group(1)
    return None


def get_prompt(model_name, company_name):
    """Generate the prompt for the API call."""
    return f"""
Please return information about the model '{model_name}' from the company '{company_name}' in strict JSON format, without extra explanation, using the following format:

{{
    "model_name": "{model_name}",
    "Licence_open_source": true or false,
    "multimodal": true or false
}}
Very important rules to follow:
- The 'Licence_open_source' field must be a boolean value: True if the model is open source, False otherwise.
- The 'multimodal' field must be a boolean value: True if the model is multimodal, False otherwise.
- Use the most recent and reliable sources for your information.
- If you don't know the answer, write null.
- Don't write any comments in the JSON.
"""


def call_perplexity_api(model_name, company_name, api_key):
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
            {"role": "user", "content": get_prompt(model_name, company_name)},
        ],
        "max_tokens": 200,  # Increased to accommodate additional information
        "temperature": 0.4,
        "top_p": 0.9,
        "return_citations": False,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response
    except requests.exceptions.RequestException as e:
        print(f"API request failed for model '{model_name}': {e}")
        return None


def call_perplexity_api_and_update(
    model_info_path, output_updated_models_path, api_key
):
    """Call the Perplexity API and update models with new information."""
    try:
        with open(model_info_path, 'r', encoding='utf-8') as file:
            model_info_data = json.load(file)

        updated_models = []

        for model_entry in model_info_data:
            model_name = model_entry.get("model_name")
            company_name = model_entry.get("company")

            if not model_name or not company_name:
                print(f"Skipping entry with missing model_name or company: {model_entry}")
                continue

            try:
                response = call_perplexity_api(model_name, company_name, api_key)

                if response and response.status_code == 200:
                    result = response.json()
                    response_text = result['choices'][0]['message']['content']

                    # Display the raw API response for diagnostic purposes
                    print(f"Raw API response for model '{model_name}': {response_text}")

                    # Extract the JSON between the ```json``` or ``` markers
                    extracted_json_str = extract_json_from_response(response_text)

                    if extracted_json_str:
                        # Clean and parse the extracted JSON content
                        cleaned_response = clean_json_response(extracted_json_str)
                        try:
                            extracted_json = json.loads(cleaned_response)
                        except json.JSONDecodeError as json_err:
                            print(f"JSON decoding failed for model '{model_name}': {json_err}")
                            continue

                        # Extraction des champs
                        licence_open_source = extracted_json.get("Licence_open_source")
                        multimodal = extracted_json.get("multimodal")

                        # Validation des champs booléens
                        if not isinstance(licence_open_source, bool) and licence_open_source is not None:
                            print(f"Invalid value for 'Licence_open_source' for model '{model_name}'. Expected a boolean.")
                            licence_open_source = None
                        if not isinstance(multimodal, bool) and multimodal is not None:
                            print(f"Invalid value for 'multimodal' for model '{model_name}'. Expected a boolean.")
                            multimodal = None

                        # Ajouter les informations extraites au modèle
                        updated_model_entry = {
                            "model_name": model_name,
                            "Licence_open_source": licence_open_source,
                            "multimodal": multimodal
                        }
                        updated_models.append(updated_model_entry)
                    else:
                        print(f"No valid JSON data found for model '{model_name}'")
                else:
                    status_code = response.status_code if response else "No response"
                    print(
                        f"Failed to fetch data for model '{model_name}', "
                        f"status code: {status_code}"
                    )

            except Exception as e:
                print(f"An error occurred while processing model '{model_name}': {e}")
                continue  # Move to the next model

        # Save the updated results to a JSON file
        with open(output_updated_models_path, 'w', encoding='utf-8') as output_file:
            json.dump(updated_models, output_file, indent=4)
        print(f"Updated model information saved to {output_updated_models_path}")

    except Exception as e:
        print(f"An error occurred while processing the model list: {e}")


def process_model_information(
    csv_path, model_info_output_path, updated_models_output_path, api_key
):
    """Main function to run the extraction and update process."""
    # Step 1: Extract model info from CSV and save to JSON
    extract_model_info_from_csv(csv_path, model_info_output_path)

    # Step 2: Call the Perplexity API and update model information
    call_perplexity_api_and_update(
        model_info_output_path, updated_models_output_path, api_key
    )


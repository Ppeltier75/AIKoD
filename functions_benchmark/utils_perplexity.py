

# Fonction pour extraire les noms de modèles depuis AIKoD_API_v16.json et les enregistrer dans un fichier JSON
def extract_model_names(json_path, output_model_names_path):
    try:
        # Charger le fichier JSON
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        
        # Extraire les noms de modèles
        model_names = [{"model_name": entry.get("model_name")} for entry in json_data if entry.get("model_name")]

        # Enregistrer les noms dans un fichier JSON
        with open(output_model_names_path, 'w') as output_file:
            json.dump(model_names, output_file, indent=4)

        print(f"Model names extracted and saved to {output_model_names_path}")

    except Exception as e:
        print(f"An error occurred while extracting model names: {e}")

# Fonction pour nettoyer les champs numériques dans les réponses JSON textuelles
def clean_json_response(json_str):
    # Remplacer "B" par "e9" pour des milliards
    json_str = re.sub(r'(\d+(\.\d+)?)\s*[Bb]', r'\1e9', json_str)
    # Remplacer "k" par "000" pour des milliers
    json_str = re.sub(r'(\d+)\s*[Kk]', lambda m: str(int(m.group(1)) * 1000), json_str)
    # Remplacer "None" (texte) par "null" (valeur JSON valide)
    json_str = json_str.replace("None", "null")
    return json_str

# Fonction pour extraire le contenu JSON entre les balises ```json``` dans la réponse API
def extract_json_from_response(response_text):
    # Chercher le contenu entre les balises ```json``` dans la réponse
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)  # Retourner la partie JSON trouvée
    return None  # Retourner None si aucun contenu valide n'a été trouvé

# Fonction pour appeler l'API Perplexity et ajouter des informations aux modèles extraits
def call_perplexity_api_and_update(model_names_path, output_updated_models_path):
    try:
        # Charger les noms de modèles depuis le fichier JSON
        with open(model_names_path, 'r') as file:
            model_names_data = json.load(file)
        
        # Sélectionner uniquement les 10 premiers modèles
        # model_names_data = model_names_data[:10]  # Ne garder que les 10 premiers modèles
        
        # Initialiser une liste pour les résultats mis à jour
        updated_models = []

        # Appeler l'API pour chaque modèle
        url = "https://api.perplexity.ai/chat/completions"
        for model_entry in model_names_data:
            model_name = model_entry.get("model_name")
            
            try:
                # Préparation de la requête API avec le prompt ajusté
                payload = {
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an API expert in data extraction and generating model details in strict JSON format."
                        },
                        {
                            "role": "user",
                            "content": f"""
                        Please return information about the model '{model_name}' in strict JSON format, without extra explanation, using the following format:

                        {{
                            "model_name": "{model_name}",
                            "model_type": "Type of model (text, image, video, multimodal)",
                            "nbr_parameters_active": "Number of active parameters (numeric only)",
                            "nbr_parameters_total": "Total number of parameters (numeric only)",
                            "context_window": "Maximum context length in tokens (numeric only)",
                            "date_release": "Release date in the format YYYY-MM-DD"
                        }}
                        Very important rules to follow : 
                        The windows context must be specified as an integer, so for example 32k becomes 32000. 
                        The number of parameters must be specified as an integer, so 32B becomes 32000000, for example. 
                        In your searches, use the results of the most recent online searches. The most reliable sources are the companies that develop the models.
                        If you don't know an answer write None, be careful not to generate hallucinations.
                        Don't write any comments in the json
                        
                        """
                        }
                    ],
                    "max_tokens": 150,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "return_citations": False,
                    "stream": False
                }

                headers = {
                    "Authorization": "Bearer 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",  # Remplacer par la clé API réelle
                    "Content-Type": "application/json"
                }

                # Effectuer la requête
                response = requests.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result['choices'][0]['message']['content']
                    
                    # Afficher la réponse brute pour diagnostic
                    print(f"Raw API response for model '{model_name}': {response_text}")
                    
                    # Extraire le JSON entre les balises ```json```
                    extracted_json_str = extract_json_from_response(response_text)
                    
                    if extracted_json_str:
                        # Nettoyer le contenu JSON extrait et le parser
                        cleaned_response = clean_json_response(extracted_json_str)
                        extracted_json = json.loads(cleaned_response)
                        
                        # Ajouter les informations extraites au modèle
                        updated_model_entry = {
                            "model_name": model_name,
                            "model_type": extracted_json.get("model_type"),
                            "nbr_parameters_active": extracted_json.get("nbr_parameters_active"),
                            "nbr_parameters_total": extracted_json.get("nbr_parameters_total"),
                            "context_window": extracted_json.get("context_window"),
                            "date_release": extracted_json.get("date_release")
                        }
                        updated_models.append(updated_model_entry)
                    else:
                        print(f"No valid JSON data found for model '{model_name}'")
                else:
                    print(f"Failed to fetch data for model '{model_name}', status code: {response.status_code}")

            # Capturer toutes les erreurs liées à un modèle spécifique pour passer au suivant
            except Exception as e:
                print(f"An error occurred while processing model '{model_name}': {e}")
                continue  # Passer au modèle suivant

        # Enregistrer les résultats mis à jour dans un fichier JSON
        with open(output_updated_models_path, 'w') as output_file:
            json.dump(updated_models, output_file, indent=4)

        print(f"Updated model information saved to {output_updated_models_path}")

    except Exception as e:
        print(f"An error occurred while processing the model list: {e}")

# Main function to run the extraction and update process
def process_model_information(original_json_path, model_names_output_path, updated_models_output_path):
    # Étape 1 : Extraire les noms de modèles et les sauvegarder dans un fichier JSON
    extract_model_names(original_json_path, model_names_output_path)

    # Étape 2 : Appeler l'API Perplexity et mettre à jour les informations des modèles
    call_perplexity_api_and_update(model_names_output_path, updated_models_output_path)




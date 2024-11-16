import re

# Fonction pour nettoyer les noms des modèles
def clean_model_name(name):
    """
    Normalizes the model name by replacing commas and hyphens with spaces,
    converting to lowercase except for numbers, and removing extra spaces.
    """
    if isinstance(name, str):  # Check if the name is a string
        name = name.lower()  # Convert to lowercase
        name = re.sub(r'[,\-]', ' ', name)  # Replace commas and hyphens with spaces
        name = re.sub(r'\s+', ' ', name).strip()  # Remove extra spaces
    return name  # Return the normalized name



from flask import Flask, request, jsonify
import os
from rdkit import Chem
from rdkit.Chem import Draw
from admet_ai import ADMETModel
import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize Flask app
app = Flask(__name__)

# Initialize Vertex AI and ADMET model
PROJECT_ID = "stellar-store-444214-e8"  # Replace with your Google Cloud project ID
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)
admet_model = ADMETModel()

# Directory to save molecule images
MOLECULE_IMAGE_DIR = "static"
os.makedirs(MOLECULE_IMAGE_DIR, exist_ok=True)

@app.route('/analyze', methods=['POST'])
def analyze_compound():
    try:
        # Get SMILES string from request
        data = request.get_json()
        smiles = data.get('smiles')
        if not smiles:
            return jsonify({"error": "SMILES string is required"}), 400

        # Generate molecule image
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol)
            img_path = os.path.join(MOLECULE_IMAGE_DIR, f"molecule_{hash(smiles)}.png")
            img.save(img_path)
        else:
            return jsonify({"error": "Invalid SMILES string"}), 400

        # Get ADMET predictions
        predictions = admet_model.predict(smiles=smiles)

        # Generate AI analysis using Vertex AI
        model = GenerativeModel("gemini-1.5-flash-002")
        prompt = f"""
        Analyze this compound:
        SMILES: {smiles}
        Properties:
        - Molecular Weight: {predictions.get('molecular_weight')}
        - logP: {predictions.get('logP')}
        - TPSA: {predictions.get('tpsa')}

        Provide a brief analysis of:
        1. Drug-likeness
        2. Potential risks
        3. Overall assessment
        """
        response = model.generate_content(prompt)

        # Return results
        return jsonify({
            "predictions": predictions,
            "analysis": response.text,
            "image_path": img_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
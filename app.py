from flask import Flask, request, jsonify
import os
from rdkit import Chem
from rdkit.Chem import Draw
from admet_ai import ADMETModel
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

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
        model = GenerativeModel("gemini-1.5-flash-002",
                                system_instruction=[
        "You are a helpful assistant for Drug Discovery.",
        "You analyze toxicity, pharmacokinetic, and safety properties of chemical compounds."
    ],)
        # Set model parameters
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192,
        )

        # Set safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        preds = admet_model.predict(smiles=smiles)
        prompt = f"""
        Analyze this compound:
        SMILES: {smiles}
        Properties:
        - Molecular Weight: {predictions.get('molecular_weight')}
        - logP: {predictions.get('logP')}
        - TPSA: {predictions.get('tpsa')}
        Hydrogen Bond Acceptors: {preds['hydrogen_bond_acceptors']}
        Hydrogen Bond Donors: {preds['hydrogen_bond_donors']}
        Lipinski Rule of 5 Compliance: {preds['Lipinski']}
        Quantitative Estimate of Drug-likeness (QED): {preds['QED']}
        Total Polar Surface Area (TPSA): {preds['tpsa']}
        AMES Toxicity: {preds['AMES']}
        Blood-Brain Barrier (BBB) Penetration: {preds['BBB_Martins']}
        Bioavailability: {preds['Bioavailability_Ma']}
        hERG Inhibition: {preds['hERG']}
        Liver Toxicity (DILI): {preds['DILI']}
        Carcinogenicity: {preds['Carcinogens_Lagunin']}
        CYP Inhibition (1A2, 2C19, 2C9, 2D6, 3A4): 
          - CYP1A2: {preds['CYP1A2_Veith']}
          - CYP2C19: {preds['CYP2C19_Veith']}
          - CYP2C9: {preds['CYP2C9_Veith']}
          - CYP2D6: {preds['CYP2D6_Veith']}
          - CYP3A4: {preds['CYP3A4_Veith']}
        Based on these properties, please provide a comprehensive toxicity and pharmacokinetic profile. Specifically include:
        1. **General Toxicity**: Acute or chronic risks based on AMES, DILI, and Carcinogenicity predictions.
        2. **Organ-Specific Toxicity**: Focus on liver (DILI), heart (hERG inhibition), and brain (BBB penetration).
        3. **Pharmacokinetics**: Analyze bioavailability, logP, and TPSA.
        4. **Drug-Likeness**: Comment on compliance with Lipinski's rule and QED score.
        5. **CYP Inhibition**: Explain risks of CYP enzyme interactions.
        6. **Overall Assessment**: Summarize safety, toxicity, and usability of the compound as a drug candidate.

        Ensure the response is clear and well-structured.
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

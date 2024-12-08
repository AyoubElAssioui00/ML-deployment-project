from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from transformers import pipeline
import pickle
import os
import numpy as np

# Configuration de l'application Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploaded_files'

# Initialisation des modèles
vgg16_model = VGG16(weights='imagenet')
text_gen_pipeline = pipeline('text-generation', model='gpt2')

# Route pour l'accueil
@app.route('/home', methods=['GET','POST'])
def home():
    return """
    <h1>Application Flask pour le Déploiement de Modèles IA</h1>
    <p>Routes disponibles :</p>
    <ul>
        <li><a href="/predict_page">Prédiction avec VGG16</a></li>
        <li><a href="/regpredict">Prédiction avec un modèle de régression</a></li>
        <li><a href="/textgen">Génération de texte avec GPT2</a></li>
    </ul>
    """



@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Sécuriser le fichier et sauvegarder temporairement
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Préparer l'image pour le modèle
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Prédire avec le modèle VGG16
            predictions = vgg16_model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=3)[0]

            # Convertir les prédictions en un format sérialisable
            serialized_predictions = [
                {"class_id": pred[0], "class_name": pred[1], "score": float(pred[2])}
                for pred in decoded_predictions
            ]

        finally:
            # Supprimer le fichier temporaire après traitement
            os.remove(file_path)

        return jsonify({'predictions': serialized_predictions})



# Route pour le modèle de régression (Pickle)
@app.route('/regpredict', methods=['GET','POST'])
def regpredict():
    input_data = request.json.get('input_data')
    if not input_data:
        return jsonify({'error': 'No input data provided'}), 400
    with open('/home/assioui/devoir3/models/regression_model.pkl', 'rb') as f:
        regression_model = pickle.load(f)
    prediction = regression_model.predict([input_data])
    return jsonify({'prediction': prediction.tolist()})

# Route pour la génération de texte
@app.route('/textgen', methods=['GET','POST'])
def textgen():
    input_text = request.json.get('text')
    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400
    result = text_gen_pipeline(input_text, max_length=50, num_return_sequences=1)
    return jsonify({'generated_text': result[0]['generated_text']})

# Lancer l'application
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

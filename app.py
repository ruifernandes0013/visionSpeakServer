from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST"], headers=["Content-Type"])
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image):
    i_image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

@app.route('/', methods=['GET'])
def ping():
    return 'API is running...'
    
@app.route('/api/photo', methods=['POST'])
def predict_photo():
    try:
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']

        # If the user does not select a file, the browser may submit an empty file without a filename
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # If the file exists and is an image
        if file:
            # Read the image file
            contents = file.read()
            
            # Open image from bytes
            image = Image.open(BytesIO(contents))
            
            # Predict caption for the image
            result = predict_step(image)
            
            # Return the predicted caption as JSON response
            return result[0], 200
        else:
            return jsonify({"error": "Invalid file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=3002)

from flask import Flask, request, jsonify
import io
from PIL import Image
from flask import Flask, request, render_template
from utils import util

app = Flask(__name__)


def caption_from_imagenet(image):
    threshold = 0.2
    img, prediction = util.get_prediction(image)
    print("prediction : ", prediction)
    top_pred = [pred[1] for pred in prediction[0] if pred[2] > threshold][:2]

    if not top_pred:
        top_pred = util.get_caption(image) 
    gen_caption = util.generate_sentence(top_pred, max_new_tokens=20)
    return gen_caption

@app.route('/', methods=['GET', 'POST'])
def caption_image():
    if request.method == "GET":
        return render_template('index.html')

    if 'image' not in request.files:
        return jsonify({'caption': 'No image provided'}), 400

    image_file = request.files['image']
    
    caption = caption_from_imagenet(image=image_file.stream)

    return jsonify({'caption': caption})
    


if __name__ == '__main__':
    app.run(debug=True, port="8080")

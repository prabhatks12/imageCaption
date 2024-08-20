from copy import deepcopy
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import keras
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image

model = VGG16(weights='imagenet')

train_df = pd.read_parquet("output/caption_data.parquet")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

def generate_sentence(words, max_new_tokens=50):
    input_text = " ".join(words)
    encoding = tokenizer.encode_plus(
        input_text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=100,  # This is the length of input tokens
        return_attention_mask=True
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Move to device if needed
    input_ids = input_ids.to(gpt_model.device)
    attention_mask = attention_mask.to(gpt_model.device)
    
    with torch.no_grad():
        output_ids = gpt_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,  # Generate up to 50 new tokens
            pad_token_id=pad_token_id
        )
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("output_text :", output_text)
    return output_text

def extract_weights_from_image(image_array):
    last_conv_layer_name = "fc2"

    model_without_last_step = tf.keras.Model(
        model.inputs, model.get_layer(last_conv_layer_name).output
    )

    # Load and preprocess the image
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Get the output of the last convolutional layer
    last_conv_layer_output = model_without_last_step.predict(img_array)
    
    # Convert numpy array to a Python list
    return last_conv_layer_output.flatten().tolist()


def get_caption(image_array):
    def calculate_similarity(row, input_image_weights):
        weights_vector = np.array(row['weights']).reshape(1, -1)
        similarity = cosine_similarity(input_image_weights, weights_vector)[0][0]
        return similarity
    
    mg = Image.open(image_array)
    img = img.resize((224, 224)) #target size of (224, 224)
    img_array = np.array(img)

    input_image_weights = np.array(extract_weights_from_image(img_array)).reshape(1, -1)
    
    data_copy = deepcopy(train_df)
    data_copy['similarity'] = data_copy.apply(lambda row : calculate_similarity(row, input_image_weights), axis=1)
    closest_match = data_copy.loc[data_copy['similarity'].idxmax()]
    closest_caption = closest_match['caption']
    return img, closest_caption

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224)) #target size of (224, 224)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add an extra dimension for batch size
    image_array = preprocess_input(img_array)
    return img, image_array

def get_prediction(image):
    img, img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    prediction = decode_predictions(prediction, top=5)
    return img, prediction
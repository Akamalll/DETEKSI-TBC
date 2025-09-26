import os
import sys
import json
import argparse
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf


def load_label_map(labels_path: str) -> dict:
	with open(labels_path, 'r') as f:
		label_map = json.load(f)
	return {int(k): v for k, v in label_map.items()}


def predict_image(model_path: str, labels_path: str, img_path: str, img_size=(224, 224)) -> None:
	model = tf.keras.models.load_model(model_path)
	label_map = load_label_map(labels_path)

	img = image.load_img(img_path, target_size=img_size)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	prob = float(model.predict(x)[0, 0])
	cls = 1 if prob >= 0.5 else 0
	label = label_map.get(cls, str(cls))
	print(f"Label: {label} | Prob: {prob:.4f}")


def main():
	parser = argparse.ArgumentParser(description="Inferensi satu gambar menggunakan model MobileNetV2 TBC")
	parser.add_argument("--model", default=os.path.join("exports", "mobilenetv2_tbc.keras"), help="Path model .keras")
	parser.add_argument("--labels", default=os.path.join("exports", "labels.json"), help="Path labels.json")
	parser.add_argument("--image", required=True, help="Path gambar input")
	parser.add_argument("--size", default="224,224", help="Ukuran input, format W,H")
	args = parser.parse_args()

	w, h = map(int, args.size.split(","))
	predict_image(args.model, args.labels, args.image, img_size=(w, h))


if __name__ == "__main__":
	main()



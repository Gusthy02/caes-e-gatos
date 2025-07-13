import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 100
model = load_model("modelo_caes_gatos.keras")

def prepare(filepath):
    img_array = cv2.imread(filepath)
    resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) / 255.0
    return resized.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

root = tk.Tk()
root.title("Classificador de Cães e Gatos")
root.geometry("400x450")
root.configure(bg="#f2f2f2")

img_label = tk.Label(root, bg="#f2f2f2")
img_label.pack(pady=10)

result_label = tk.Label(root, text="Selecione uma imagem", font=("Arial", 14), bg="#f2f2f2")
result_label.pack(pady=10)

def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = ImageOps.fit(image, (250, 250), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        img_label.config(image=photo)
        img_label.image = photo

        prediction = model.predict(prepare(file_path))
        result = "🐶 Cachorro" if prediction[0][0] > 0.5 else "🐱 Gato"
        result_label.config(text=f"Previsão: {result}")

select_button = tk.Button(root, text="Selecionar Imagem", command=predict_image, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
select_button.pack(pady=20)

root.mainloop()

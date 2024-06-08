import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image, ImageOps
import cv2

# Carga y preprocesa la imagen
def load_and_preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Carga el modelo MobileNetV2 preentrenado
def is_cat(img_path):
    model = MobileNetV2(weights='imagenet')
    img_array = load_and_preprocess_image(img_path)
    # Realiza la predicción
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    # Verifica si alguna de las predicciones contiene 'cat'
    for _, label, _ in decoded_predictions:
        if 'cat' in label:
            return True
    return False

# Detecta la cara del gato y agrega bigote
def add_mustache(img_path):
    # Carga la imagen y convierte a escala de grises
    base_image = cv2.imread(img_path)
    gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    
    # Carga el clasificador de Haar
    cat_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml'
    cat_cascade = cv2.CascadeClassifier(cat_cascade_path)
    
    if cat_cascade.empty():
        print(f"Error al cargar el clasificador de Haar desde {cat_cascade_path}")
        return Image.fromarray(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
    
    # Detecta la cara del gato
    faces = cat_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Puedes ajustar este valor
        minNeighbors=5,   # Puedes ajustar este valor
        minSize=(50, 50)  # Puedes ajustar este valor
    )
    
    if len(faces) == 0:
        print("No se detectaron caras de gato.")
        return Image.fromarray(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
    
    for (x, y, w, h) in faces:
        print(f"Face detected at X: {x}, Y: {y}, Width: {w}, Height: {h}")
        # Ajusta la posición y el tamaño del bigote
        mustache = Image.open('mustache.png')
        mustache = mustache.resize((w, h // 4), Image.Resampling.LANCZOS)
        
        # Calcula la posición para colocar el bigote (debajo de la nariz)
        mustache_x = x + (w - mustache.width) // 2
        mustache_y = y + (h // 2)
        
        # Convierte la imagen base a PIL para pegar el bigote
        base_pil_image = Image.fromarray(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
        base_pil_image.paste(mustache, (mustache_x, mustache_y), mustache)
        return base_pil_image

if __name__ == "__main__":
    # Ruta de la imagen a analizar
    img_path = 'gato.jpg'
    # Verifica si la imagen contiene un gato
    if is_cat(img_path):
        print("La imagen contiene un gato.")
        result_image = add_mustache(img_path)
        result_image.show()  # Muestra la imagen con el bigote
        result_image.save('cat_with_mustache.jpg')  # Guarda la imagen con el bigote
    else:
        print("La imagen no contiene un gato.")

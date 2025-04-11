import os
import numpy as np
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gzip
import shutil


# Preprocess image function
def preprocess_image(image_path, target_size):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Load models
diapath = os.path.join(settings.BASE_DIR, 'predictor/models')
parkinson_model = load_model(os.path.join(diapath, 'parkinson_model.h5'))
diabetic_retinopathy_model = load_model(os.path.join(diapath, 'dr_model.h5'))
cardiomegaly_model = load_model(os.path.join(diapath, 'cardiomegaly_model.h5'))

# Views
def home(request):
    return render(request, 'home.html')

def parkinson(request):
    result = None
    error = None

    if request.method == "POST" and 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        fs = FileSystemStorage(location=upload_dir)
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_file_path = fs.path(file_path)

        try:
            image_array = preprocess_image(full_file_path, target_size=(128, 128))
            prediction = parkinson_model.predict(image_array)
            result = "Positive for Parkinson's" if prediction[0][0] > 0.5 else "Negative for Parkinson's"
        except Exception as e:
            error = f"An error occurred while processing the image: {str(e)}"
        finally:
            if os.path.exists(full_file_path):
                os.remove(full_file_path)

    # Change 'index.html' to 'parkinson.html'
    return render(request, 'parkinson.html', {'result': result, 'error': error})

def decompress_model(gz_file, output_file):
    if not os.path.exists(output_file):  # Decompress only if not already decompressed
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

# Path to the compressed models
compressed_models = {
    "parkinson": "predictor/models/parkinson_model.h5.gz",
}

# Decompress and load the models
for key, gz_path in compressed_models.items():
    output_path = gz_path.replace(".gz", "")  # Remove .gz to get the model file name
    decompress_model(gz_path, output_path)

# Load models after decompression
parkinson_model = load_model("predictor/models/parkinson_model.h5")




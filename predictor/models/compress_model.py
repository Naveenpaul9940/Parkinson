import gzip
import shutil

# Paths to the original and compressed files
original_file = "parkinson_model.h5"
compressed_file = "parkinson_model.h5.gz"

# Compress the file
with open(original_file, 'rb') as f_in:
    with gzip.open(compressed_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"File compressed and saved as: {compressed_file}")

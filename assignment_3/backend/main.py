from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
import librosa
import cv2
import numpy as np

from processors.text_processor import TextProcessor
from processors.image_processor import ImageProcessor
from processors.audio_processor import AudioProcessor
from utils.file_utils import get_file_type, ensure_directories

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
STATIC_DIR = "static"
SUBDIRS = ["uploads", "processed", "augmented"]
paths = ensure_directories(STATIC_DIR, SUBDIRS)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize processors
text_processor = TextProcessor()
image_processor = ImageProcessor()
audio_processor = AudioProcessor()

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    # Save original file
    file_path = paths["uploads"] / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_type = get_file_type(file.filename)
    
    if file_type == 'text':
        # Read the text content
        with open(file_path, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        # Process text
        preprocessed_text = text_processor.preprocess(original_text)
        augmented_text, techniques = text_processor.augment(preprocessed_text)
        
        # Save processed and augmented versions
        processed_path = paths["processed"] / file.filename
        augmented_path = paths["augmented"] / file.filename
        
        with open(processed_path, 'w', encoding='utf-8') as f:
            f.write(preprocessed_text)
        
        with open(augmented_path, 'w', encoding='utf-8') as f:
            f.write(augmented_text)
        
        return JSONResponse({
            "message": "File processed successfully",
            "file_type": file_type,
            "urls": {
                "original": f"/static/uploads/{file.filename}",
                "processed": f"/static/processed/{file.filename}",
                "augmented": f"/static/augmented/{file.filename}"
            },
            "augmentation_info": {
                "techniques_applied": techniques
            }
        })
        
    elif file_type == 'image':
        # Define paths
        processed_path = paths["processed"] / file.filename
        augmented_path = paths["augmented"] / file.filename
        
        try:
            # Preprocess
            preprocessed_image = image_processor.preprocess(str(file_path))
            
            # Save preprocessed image
            image_processor.save_image(preprocessed_image, processed_path)
            
            # Augment the preprocessed image
            augmented_image, techniques = image_processor.augment(preprocessed_image)
            
            # Save augmented image
            image_processor.save_image(augmented_image, augmented_path)
            
            return JSONResponse({
                "message": "File processed successfully",
                "file_type": file_type,
                "urls": {
                    "original": f"/static/uploads/{file.filename}",
                    "processed": f"/static/processed/{file.filename}",
                    "augmented": f"/static/augmented/{file.filename}"
                },
                "augmentation_info": {
                    "techniques_applied": techniques
                }
            })
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return JSONResponse({
                "error": "Error processing image",
                "details": str(e)
            }, status_code=500)
            
    elif file_type == 'audio':
        # Define paths for audio and spectrograms
        processed_path = paths["processed"] / file.filename
        augmented_path = paths["augmented"] / file.filename
        
        # Define spectrogram paths
        orig_spec_path = paths["uploads"] / f"{file.filename}_spec.png"
        proc_spec_path = paths["processed"] / f"{file.filename}_spec.png"
        aug_spec_path = paths["augmented"] / f"{file.filename}_spec.png"
        
        try:
            # Preprocess
            audio, sr = audio_processor.preprocess(str(file_path))
            
            # Save preprocessed audio and its spectrogram
            audio_processor.save_audio(audio, sr, processed_path)
            audio_processor.create_spectrogram(audio, sr, proc_spec_path)
            
            # Create spectrogram for original audio
            orig_audio, orig_sr = librosa.load(str(file_path), sr=None)
            audio_processor.create_spectrogram(orig_audio, orig_sr, orig_spec_path)
            
            # Augment
            augmented_audio, techniques = audio_processor.augment(audio, sr)
            
            # Save augmented audio and its spectrogram
            audio_processor.save_audio(augmented_audio, sr, augmented_path)
            audio_processor.create_spectrogram(augmented_audio, sr, aug_spec_path)
            
            return JSONResponse({
                "message": "File processed successfully",
                "file_type": file_type,
                "urls": {
                    "original": f"/static/uploads/{file.filename}",
                    "original_spec": f"/static/uploads/{file.filename}_spec.png",
                    "processed": f"/static/processed/{file.filename}",
                    "processed_spec": f"/static/processed/{file.filename}_spec.png",
                    "augmented": f"/static/augmented/{file.filename}",
                    "augmented_spec": f"/static/augmented/{file.filename}_spec.png"
                },
                "augmentation_info": {
                    "techniques_applied": techniques
                }
            })
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return JSONResponse({
                "error": "Error processing audio",
                "details": str(e)
            }, status_code=500)
    
    # Handle unsupported file types
    return JSONResponse({
        "error": "Unsupported file type",
        "details": f"File type '{file_type}' is not supported"
    }, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
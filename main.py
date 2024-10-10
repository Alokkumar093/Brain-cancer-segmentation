from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import nibabel as nib
import numpy as np
import os

from utils.brain_tumor import BrainTumorModel

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "(link unavailable)",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


brain_model = BrainTumorModel()

@app.post("/predict", responses={
    200: {
        "description": "A gif showing the MRI slices with segmentation overlay.",
        "content": {
            "image/gif": {
                "schema": {
                    "type": "string",
                    "format": "binary"
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error"
    }
})
async def predict(flair: UploadFile = File(...), t1ce: UploadFile = File(...)):
    """
    Make a prediction on brain tumor segmentation.
    
    - **flair**: Nifti file for flair MRI.
    - **t1ce**: Nifti file for t1ce MRI.
    """
    # Save files to disk
    flair_pth = f'data/{flair.filename}'
    t1ce_pth = f'data/{t1ce.filename}'
    
    try:
        with open(flair_pth, "wb") as f:
            f.write(await flair.read())
        with open(t1ce_pth, "wb") as f:
            f.write(await t1ce.read())
        
        # Load NIfTI files
        flair_data = nib.load(flair_pth).get_fdata()
        t1ce_data = nib.load(t1ce_pth).get_fdata()
        
        # Make prediction
        prediction = brain_model.make_prediction(flair_data, t1ce_data)
        mri_data = np.argmax(prediction, axis=-1)
        
        # Create GIF
        gif_filename = brain_model.create_gif(flair_data, mri_data)
        
        # GIF generation
        response = FileResponse(gif_filename, media_type='image/gif')

        # Delete files after sending response
        os.remove(flair_pth)
        os.remove(t1ce_pth)
        # os.remove(gif_filename)

        return response

    except Exception as e:
        # Clean up files in case of error
        if os.path.exists(flair_pth):
            os.remove(flair_pth)
        if os.path.exists(t1ce_pth):
            os.remove(t1ce_pth)
        raise HTTPException(status_code=500, detail=f"Error happened: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from model_inference import predict_emotion

app = FastAPI()

# CORS config for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # "*" means all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        print(f"Received file: {file.filename}")
        emotion = predict_emotion(image)
        
        return {"emotion": emotion}
    except Exception as e:
        return {"error": str(e)}


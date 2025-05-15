from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from src.config import APP_NAME, VERSION, SECRET_KEY_TOKEN, MODEL_PATH
from src.inference import load_model, predict_image
from PIL import Image
import io
import uvicorn

app = FastAPI(title=APP_NAME, version=VERSION)

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != SECRET_KEY_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized API access")
    return api_key

# Load model at startup
model = load_model(MODEL_PATH)

@app.get("/", tags=["Health"], description="Health Check")
async def home(api_key: str = Depends(verify_api_key)):
    return {"message": f"Welcome to {APP_NAME} API v{VERSION}"}

@app.post("/predict", tags=["Prediction"], description="Make prediction using the model")
async def predict(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        predictions = predict_image(image, model)
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
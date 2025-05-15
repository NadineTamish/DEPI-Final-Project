from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from src.config import APP_NAME, VERSION, SECRET_KEY_TOKEN, MODEL_PATH
from src.inference import load_model, predict_image
from PIL import Image
import io
import os
import shutil
import uuid

app = FastAPI(title=APP_NAME, version=VERSION)

# Mount static files (for CSS and uploaded images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template directory
templates = Jinja2Templates(directory="templates")

# API Key Auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != SECRET_KEY_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized API access")
    return api_key

# Load model at startup
model = load_model(MODEL_PATH)

# Route to render index.html
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "filename": None})

# Form submission with file upload
@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
):
    try:
        # Save uploaded file to static/uploads/
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Unique filename to avoid cache issues
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        image.save(filepath)

        # Run inference
        predictions = predict_image(image, model)

        # Pass data to template
        return templates.TemplateResponse("result.html", {
            "request": request,
            "user_image": f"uploads/{filename}",
            "detected_objects": predictions
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Optional: Simple JSON health check route
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

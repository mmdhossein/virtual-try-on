# backend/main.py

import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import httpx
import uuid
from pathlib import Path
import logging
import asyncio
import json
from PIL import Image
from fastapi.responses import Response

from .catvton_service import run_tryon

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fashion Try-On API Gateway")

# Directories
UPLOAD_DIR = "temp"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "http://172.21.0.1/v1/workflows/run")

TIMEOUT = float(os.getenv("TIMEOUT", "240.0")) # Timeout(timeout=5.0)
USE_DUMMY_DIFY = os.getenv("USE_DUMMY_DIFY", "false").lower() == "true"
DIFY_AUTH_TOKE = os.getenv("DIFY_AUTH_TOKE", "Bearer app-2zm5vMbFKPgPkO9vNTvZXaAk")

# In-memory storage (use Redis/DB in production)
image_store: Dict[str, bytes] = {}  
sessions: Dict[str, Dict] = {}

# ============================================================================
# Request Models
# ============================================================================


class ChatRequest(BaseModel):
    session_id: str
    message: str
    person_image_id: Optional[str] = None
    cloth_image_id: Optional[str] = None
    chat_history: List[Dict[str, str]] = []
    last_output_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    tryon_ready: bool = False


class TryOnRequest(BaseModel):
    session_id: str
    person_image_id: str
    cloth_image_id: str
    cloth_type: str = "full"  # # ["upper", "lower", "full"],


class TryOnResponse(BaseModel):
    output_image_id: str
    image_base64: str


class UploadResponse(BaseModel):
    image_id: str
    detected_type: str
    message: str


# ============================================================================
# Dummy Dify Service (for testing without actual Dify)
# ============================================================================

if USE_DUMMY_DIFY:
    logger.warning("⚠️  DUMMY DIFY MODE ENABLED - Using simulated responses")


async def forward_to_dify_dummy(
    json_data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    timeout: float = TIMEOUT,
) -> httpx.Response:
    """
    Dummy Dify service for testing without actual Dify backend.
    Simulates classification and chat responses.
    """
    from unittest.mock import Mock

    # Simulate network delay
    await asyncio.sleep(0.3)

    action = json_data.get("action") if json_data else None

    # Handle classification
    if action == "classify":
        filename = ""
        if files and "file" in files:
            # Extract filename from tuple (filename, content, content_type)
            filename = (
                files["file"][0].lower() if isinstance(files["file"], tuple) else ""
            )
        detected_type = (
            "person" if "person" in filename or "model" in filename else "cloth"
        )

        response_data = {
            "detected_type": detected_type,
            "message": f"✓ Image classified as {detected_type}. {'I got your photo!' if detected_type == 'person' else 'Great! Received the cloth.'}",
        }

        logger.info(f"[DUMMY] Classified image as: {detected_type}")

    # Handle chat
    elif action == "chat":
        message = json_data.get("message", "").lower()
        person_id = json_data.get("person_image_id")
        cloth_id = json_data.get("cloth_image_id")
        session_id = json_data.get("session_id")
        last_output_id = json_data.get("last_output_id")

        # Check if both images are present
        both_present = person_id and cloth_id

        # Determine user intent
        tryon_keywords = ["try", "show", "see", "wear", "put on", "how", "look"]
        suggest_keywords = [
            "suggest",
            "accessory",
            "accessories",
            "match",
            "goes with",
            "recommend",
        ]
        wants_tryon = any(keyword in message for keyword in tryon_keywords)
        wants_suggestions = any(keyword in message for keyword in suggest_keywords)

        # Check if try-on result exists
        has_tryon_result = (
            session_id in sessions and sessions[session_id].get("tryon_id") is not None
        )

        # Generate response based on context
        if wants_suggestions and has_tryon_result:
            # Simulate accessory suggestion based on try-on result
            reply = (
                "Based on your outfit, I'd suggest:\n"
                "• A silver minimalist watch\n"
                "• White leather sneakers\n"
                "• A delicate gold necklace\n"
                "These would complement the look perfectly! ✨"
            )
            tryon_ready = False
            logger.info(
                f"[DUMMY] Providing accessory suggestions for session {session_id}"
            )

        elif wants_suggestions and not has_tryon_result:
            reply = "I'd love to suggest accessories! First, let's see how the outfit looks on you. Would you like to try it on?"
            tryon_ready = False

        elif both_present and wants_tryon:
            reply = "Perfect! I have both your photo and the clothing item. Let me show you how it looks! 👗✨"
            tryon_ready = True
            logger.info(f"[DUMMY] Triggering try-on for session {session_id}")

        elif both_present:
            reply = "I see you have both images uploaded! Would you like to try on the outfit? Just say 'show me' or 'try it on'!"
            tryon_ready = False

        elif person_id and not cloth_id:
            reply = (
                "I have your photo! Please upload a clothing item you'd like to try on."
            )
            tryon_ready = False

        elif cloth_id and not person_id:
            reply = "I have the clothing item! Please upload your photo so we can see how it looks on you."
            tryon_ready = False

        else:
            reply = "Hi! I'm your virtual fashion assistant. Upload your photo and a clothing item to get started! 👋"
            tryon_ready = False

        response_data = {"reply": reply, "tryon_ready": tryon_ready}

    else:
        response_data = {
            "error": "Unknown action",
            "message": "Please specify 'classify' or 'chat' action",
        }
        logger.warning(f"[DUMMY] Unknown action: {action}")

    # Create mock response
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": {"outputs": response_data}}
    mock_response.raise_for_status = Mock()

    return mock_response


# ============================================================================
# Helper Functions
# ============================================================================


async def forward_to_dify(
    json_data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    timeout: float = TIMEOUT,
) -> httpx.Response:
    """Forward request to Dify webhook or use dummy service for testing."""

    # Use dummy service if enabled
    if USE_DUMMY_DIFY:
        return await forward_to_dify_dummy(json_data, files, timeout)

    # Real Dify forwarding logic
    url = DIFY_BASE_URL

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # if files:
            # form-data is not supported on dify
            # For file uploads, use multipart/form-data
            #     response = await client.post(url, data=json_data, files=files)
            # else:
            # For JSON-only requests

            json_data = {
                "inputs": json_data,
                "response_mode": "blocking",
                "user": "api-service",
            }
            print("----------Request--------\n", "url: ", url, "body: ", json_data)
            response = await client.post(
                url, json=json_data, headers={"Authorization": DIFY_AUTH_TOKE}
            )
            print("----------response--------\n", response.json())

            response.raise_for_status()
            logger.info(
                f"Dify request successful: {json_data.get('action', 'unknown')}"
            )
            return response

        except httpx.TimeoutException:
            logger.error(f"Timeout calling Dify (>{timeout}s)")
            raise HTTPException(status_code=504, detail="Dify service timeout")
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error from Dify: {e.response.status_code} - {e.response.text}"
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Dify service error: {e.response.text}",
            )
        except Exception as e:
            logger.error(f"Unexpected error calling Dify: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def get_image_base64(image_id: str) -> str:
    """Get base64 encoded image from store."""
    if image_id not in image_store:
        raise HTTPException(status_code=404, detail="Image not found")
    return base64.b64encode(image_store[image_id]).decode()


def get_image_base64_v2(
    image_id: str, max_size: tuple = (1024, 1024), quality: int = 85
) -> str:
    """Get base64 encoded image from store with compression."""
    if image_id not in image_store:
        raise HTTPException(status_code=404, detail="Image not found")

    # Load image from bytes
    image_bytes = image_store[image_id]
    img = Image.open(io.BytesIO(image_bytes))

    # Convert RGBA to RGB if necessary (for JPEG compatibility)
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(
            img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None
        )
        img = background

    # Resize if image is larger than max_size (maintains aspect ratio)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Compress to JPEG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode()


# ============================================================================
# API Endpoints
# ============================================================================


@app.post("/api/upload", response_model=UploadResponse)
async def upload_image(session_id: str = Form(...), file: UploadFile = File(...)):
    """
    Upload image and let Dify classify it as person or cloth.
    Frontend expects: {image_id, detected_type, message}
    """
    try:
        # Generate unique image ID
        image_id = str(uuid.uuid4())

        # Read and store file content
        content = await file.read()
        image_store[image_id] = content

        # Initialize session if needed
        if session_id not in sessions:
            sessions[session_id] = {
                "person_image_id": None,
                "cloth_image_id": None,
                "chat_history": [],
                "tryon_id": None,
            }

        # Prepare file for Dify classification
        files = {
            "file": (
                file.filename or "image.png",
                io.BytesIO(content),
                file.content_type or "image/png",
            )
        }

        payload = {
            "session_id": session_id,
            "image_id": image_id,
            "action": "classify",
            "person_image_id": sessions[session_id]["person_image_id"],
            "cloth_image_id": sessions[session_id]["cloth_image_id"],
        }

        # Forward to Dify for classification
        response = await forward_to_dify(json_data=payload, files=files)  # files=files
        dify_result = response.json()
        dify_result = dify_result.get("data").get("outputs")
        # Extract classification from Dify response
        detected_type = dify_result.get("detected_type", "unknown")
        message = dify_result.get("reply", f"Image classified as {detected_type}")

        # Update session with the classified image
        if detected_type == "person":
            sessions[session_id]["person_image_id"] = image_id
        elif detected_type == "cloth":
            sessions[session_id]["cloth_image_id"] = image_id

        logger.info(
            f"Image {image_id} uploaded and classified as {detected_type} for session {session_id}"
        )

        return UploadResponse(
            image_id=image_id, detected_type=detected_type, message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - forwards to Dify to determine if try-on should be triggered.
    Frontend expects: {reply, tryon_ready}
    """
    try:
        # Initialize session if needed
        if request.session_id not in sessions:
            sessions[request.session_id] = {
                "person_image_id": request.person_image_id,
                "cloth_image_id": request.cloth_image_id,
                "chat_history": [],
                "tryon_id": None,
            }

        tryon_id = sessions[request.session_id].get("tryon_id")

        # Prepare payload for Dify
        payload = {
            "session_id": request.session_id,
            "message": request.message,
            "person_image_id": request.person_image_id,
            "cloth_image_id": request.cloth_image_id,
            "chat_history": json.dumps(request.chat_history or []),  # Array to string
            "last_output_id": request.last_output_id,
            "action": "chat",
            "tryon_id": tryon_id,
        }

        # Forward to Dify
        response = await forward_to_dify(json_data=payload)
        result = response.json()
        result = result.get("data").get("outputs")
        # Extract response
        reply = result.get("reply", "I'm here to help with virtual try-on!")
        tryon_ready = result.get("tryon_ready", False)

        # Update session chat history
        sessions[request.session_id]["chat_history"].append(
            {"role": "user", "content": request.message}
        )
        sessions[request.session_id]["chat_history"].append(
            {"role": "assistant", "content": reply}
        )

        logger.info(
            f"Chat processed for session {request.session_id}, tryon_ready={tryon_ready}"
        )

        return ChatResponse(reply=reply, tryon_ready=tryon_ready)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/api/tryon", response_model=TryOnResponse)
async def tryon(request: TryOnRequest):
    """
    Virtual try-on endpoint using CatVTON.
    Frontend expects: {output_image_id, image_base64}
    """
    try:
        # Validate images exist
        if request.person_image_id not in image_store:
            raise HTTPException(status_code=404, detail="Person image not found")
        if request.cloth_image_id not in image_store:
            raise HTTPException(status_code=404, detail="Cloth image not found")

        # Generate unique ID for this try-on
        uid = str(uuid.uuid4())

        # Save images to temp files
        person_path = f"{UPLOAD_DIR}/{uid}_person.png"
        cloth_path = f"{UPLOAD_DIR}/{uid}_cloth.png"

        with open(person_path, "wb") as f:
            f.write(image_store[request.person_image_id])

        with open(cloth_path, "wb") as f:
            f.write(image_store[request.cloth_image_id])

        # Run CatVTON
        logger.info(f"Running try-on for session {request.session_id}")
        result_image = run_tryon(
            person_path=person_path,
            cloth_path=cloth_path,
            cloth_type=request.cloth_type,
            num_inference_steps=50,
            guidance_scale=4,
            seed=42,
            show_type="result only",
        )

        # Convert result to bytes
        img_bytes = io.BytesIO()
        result_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        result_data = img_bytes.getvalue()

        # Generate output image ID
        output_image_id = str(uuid.uuid4())

        # Store result
        image_store[output_image_id] = result_data

        # Update session with try-on result
        if request.session_id in sessions:
            sessions[request.session_id]["tryon_id"] = output_image_id
            logger.info(
                f"Stored try-on result {output_image_id} for session {request.session_id}"
            )
        else:
            logger.warning(
                f"Session {request.session_id} not found when storing try-on result"
            )

        # Convert to base64 for frontend
        image_base64 = base64.b64encode(result_data).decode()

        # Clean up temp files
        try:
            os.remove(person_path)
            os.remove(cloth_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temp files: {cleanup_error}")

        logger.info(f"Try-on completed: {output_image_id}")

        return TryOnResponse(output_image_id=output_image_id, image_base64=image_base64)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Try-on error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Try-on failed: {str(e)}")


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    """Retrieve an image by ID."""
    if image_id not in image_store:
        raise HTTPException(status_code=404, detail="Image not found")

    return JSONResponse(
        content={"image_id": image_id, "image_base64": get_image_base64_v2(image_id)}
    )


@app.get("/api/image/{image_id}/preview")
async def get_image_preview(image_id: str):
    """Get resized image as JPEG (for browser viewing)."""
    if image_id not in image_store:
        raise HTTPException(status_code=404, detail="Image not found")

    image_bytes = image_store[image_id]
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(
            img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None
        )
        img = background

    # Resize
    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    # Return as JPEG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85, optimize=True)
    buffer.seek(0)

    return Response(content=buffer.read(), media_type="image/jpeg")


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session data."""
    if session_id in sessions:
        # Clean up associated images
        session = sessions[session_id]
        for key in ["person_image_id", "cloth_image_id", "tryon_id"]:
            img_id = session.get(key)
            if img_id and img_id in image_store:
                del image_store[img_id]

        del sessions[session_id]
        logger.info(f"Session {session_id} cleared")

    return {"message": "Session cleared successfully"}


@app.get("/api/session/all")
async def get_all_sessions():

    return {"sessions": sessions}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Fashion Try-On API Gateway",
        "dify_mode": "dummy" if USE_DUMMY_DIFY else "real",
        "sessions": len(sessions),
        "images": len(image_store),
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Fashion Try-On API Gateway",
        "version": "1.0.0",
        "dify_mode": "dummy" if USE_DUMMY_DIFY else "real",
        "endpoints": {
            "upload": "POST /api/upload - Upload and classify image via Dify",
            "chat": "POST /api/chat - Chat and determine try-on readiness via Dify",
            "tryon": "POST /api/tryon - Execute virtual try-on with CatVTON",
            "image": "GET /api/image/{image_id} - Retrieve stored image",
            "clear": "DELETE /api/session/{session_id} - Clear session",
            "health": "GET /health - Health check",
        },
        "dify_target": DIFY_BASE_URL if not USE_DUMMY_DIFY else "dummy service",
    }


# ============================================================================
# Startup/Shutdown Events
# ============================================================================


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Fashion Try-On API Gateway started")
    logger.info(f"Dify Mode: {'DUMMY' if USE_DUMMY_DIFY else 'REAL'}")
    if not USE_DUMMY_DIFY:
        logger.info(f"Dify URL: {DIFY_BASE_URL}")
    logger.info(f"Timeout: {TIMEOUT}s")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Fashion Try-On API Gateway shutting down")
    image_store.clear()
    sessions.clear()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

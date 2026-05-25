import gradio as gr
import requests
from PIL import Image
import io
import base64
import uuid
from typing import List, Dict, Optional

# API Configuration
API_BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/upload"
CHAT_ENDPOINT = f"{API_BASE_URL}/api/chat"
TRYON_ENDPOINT = f"{API_BASE_URL}/api/tryon"

# ============================================================================
# State Management
# ============================================================================

class SessionState:
    def __init__(self):
        self.session_id: str = str(uuid.uuid4())
        self.person_image_id: Optional[str] = None
        self.cloth_image_id: Optional[str] = None
        self.tryon_history: List[Dict] = []
        self.chat_history: List[Dict] = []
    
    def add_tryon_result(self, person_id: str, cloth_id: str, output_id: str):
        self.tryon_history.append({
            "person_id": person_id,
            "cloth_id": cloth_id,
            "output_id": output_id
        })
    
    def get_last_output_id(self) -> Optional[str]:
        if self.tryon_history:
            return self.tryon_history[-1]["output_id"]
        return None

# ============================================================================
# API Helper Functions
# ============================================================================

def upload_image(image: Image.Image, filename: str, state: SessionState) -> Optional[Dict]:
    """Upload image with session context and get classification."""
    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        files = {"file": (filename or 'image.png', img_bytes, "image/png")}
        data = {
            "session_id": state.session_id,
            "person_image_id": state.person_image_id or "",
            "cloth_image_id": state.cloth_image_id or ""
        }
        
        response = requests.post(UPLOAD_ENDPOINT, files=files, data=data, timeout=600)
        
        if response.status_code == 200:
            result = response.json()
            image_id = result.get("image_id")
            detected_type = result.get("detected_type")
            
            if detected_type == "person":
                state.person_image_id = image_id
            elif detected_type == "cloth":
                state.cloth_image_id = image_id
            
            return result
        return None
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return {"error": str(e)}


def call_chat_agent(message: str, state: SessionState) -> Dict:
    """Call chat agent with full session context."""
    try:
        payload = {
            "session_id": state.session_id,
            "message": message,
            "person_image_id": state.person_image_id,
            "cloth_image_id": state.cloth_image_id,
            "chat_history": state.chat_history,
            "last_output_id": state.get_last_output_id()
        }
        
        response = requests.post(CHAT_ENDPOINT, json=payload, timeout=600)
        
        if response.status_code == 200:
            return response.json()
        return {"reply": "Error connecting to server", "tryon_ready": False}
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return {"reply": f"Error: {str(e)}", "tryon_ready": False}


def call_tryon(state: SessionState, cloth_type: str = "upper") -> Optional[Dict]:
    """Call tryon endpoint with session context."""
    try:
        payload = {
            "session_id": state.session_id,
            "person_image_id": state.person_image_id,
            "cloth_image_id": state.cloth_image_id,
            "cloth_type": cloth_type,
            "steps": 30,
            "seed": 42
        }
        
        response = requests.post(TRYON_ENDPOINT, json=payload, timeout=600)
        
        if response.status_code == 200:
            result = response.json()
            return result
        return None
    except Exception as e:
        print(f"Tryon error: {str(e)}")
        return {"error": str(e)}


# ============================================================================
# Processing Functions
# ============================================================================

def process_message(message: str, files: Optional[List], state: SessionState):
    """Main processing function with session isolation."""
    
    # Step 1: Upload any new images
    if files:
        for file in files:
            try:
                image = Image.open(file)
                upload_result = upload_image(image, file.name, state)
                
                if upload_result and "error" not in upload_result:
                    detected_type = upload_result.get("detected_type", "image")
                    backend_msg = upload_result.get("message", f"{detected_type.capitalize()} uploaded")
                    
                    state.chat_history.append({
                        "role": "assistant",
                        "content": f"✅ {backend_msg}"
                    })
                else:
                    error_msg = upload_result.get("error", "Upload failed") if upload_result else "Upload failed"
                    state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ {error_msg}"
                    })
            except Exception as e:
                state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Failed to process image: {str(e)}"
                })
    
    # Step 2: Process text message if provided
    result_image = None
    
    if message and message.strip():
        state.chat_history.append({"role": "user", "content": message})
        
        agent_response = call_chat_agent(message, state)
        
        reply = agent_response.get("reply", "")
        tryon_ready = agent_response.get("tryon_ready", False)
        
        state.chat_history.append({"role": "assistant", "content": reply})
        
        # Step 3: If tryon_ready, execute try-on
        if tryon_ready and state.person_image_id and state.cloth_image_id:
            cloth_type = agent_response.get("cloth_type", "upper")
            tryon_result = call_tryon(state, cloth_type)
            
            if tryon_result and "image_base64" in tryon_result:
                output_image_id = tryon_result["output_image_id"]
                image_b64 = tryon_result["image_base64"]
                
                state.add_tryon_result(
                    state.person_image_id,
                    state.cloth_image_id,
                    output_image_id
                )
                
                image_data = base64.b64decode(image_b64)
                result_image = Image.open(io.BytesIO(image_data))
                
                result_msg = tryon_result.get("message", "✨چطور شد؟😊✅")
                state.chat_history.append({
                    "role": "assistant",
                    "content": result_msg
                })
                
                state.cloth_image_id = None
            
            elif tryon_result and "error" in tryon_result:
                state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Try-on failed: {tryon_result['error']}"
                })
    
    chat_display = format_chat_history(state.chat_history)
    
    return chat_display, result_image, state


# def format_chat_history(history: List[Dict]) -> str:
#     """Format chat history as messaging app bubbles."""
#     if not history:
#         return '<div class="welcome-msg">👋 Welcome! Upload your photo and a clothing item, then ask me to try it on you!</div>'
    
#     formatted = []
#     for msg in history:
#         role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
#         content = msg["content"].replace("\n", "<br>")
#         formatted.append(f'<div class="msg-bubble {role_class}">{content}</div>')
    
#     # return "".join(formatted)
#     return f'<div style="max-height: 600px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px;">{"".join(formatted)}</div>'


def format_chat_history(history: List[Dict]) -> str:
    """Format chat history as messaging app bubbles."""
    if not history:
        return '<div class="welcome-msg">👋 Welcome! Upload your photo and a clothing item, then ask me to try it on you!</div>'
    
    formatted = []
    for msg in history:
        role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
        content = msg["content"].replace("\n", "<br>")
        formatted.append(f'<div class="msg-bubble {role_class}">{content}</div>')
    
    return f'''
    <div style="
        max-height: 100%;
        height: 100%;
        overflow-y: auto;
        overflow-x: hidden;
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding-right: 8px;
        padding-bottom: 20px;
        box-sizing: border-box;
    ">
        {"".join(formatted)}
    </div>
    '''


def clear_session(state: SessionState):
    """Clear session and reset state."""
    state.chat_history = []
    state.person_image_id = None
    state.cloth_image_id = None
    state.tryon_history = []
    
    return format_chat_history([]), None, state


# ============================================================================
# Custom CSS
# ============================================================================

custom_css = """
/* Main layout */
#main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
    background: #0f0f14;
    color: #e6e6f0;
}

/* Chat display container */
#chat-display {
    min-height: 400px;
    max-height: 600px;
    height: 600px;  /* Fixed height */
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(180deg, #16161d 0%, #121218 100%);
    border: 1px solid #262637;
    overflow: auto !important;  /* Prevent outer overflow */
    box-sizing: border-box;
}

/* Message bubbles */
.msg-bubble {
    padding: 12px 16px;
    border-radius: 16px;
    max-width: 80%;
    word-wrap: break-word;
    overflow-wrap: break-word;
    word-break: break-word;  /* Force long words to break */
    box-sizing: border-box;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* User messages - right aligned, blue */
.user-msg {
    align-self: flex-end;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-bottom-right-radius: 4px;
    margin-left: auto;
}

/* Assistant messages - left aligned, dark */
.assistant-msg {
    align-self: flex-start;
    background: #1e1e2e;
    color: #e6e6f0;
    border: 1px solid #2a2a3d;
    border-bottom-left-radius: 4px;
    margin-right: auto;
}

/* Welcome message */
.welcome-msg {
    text-align: center;
    color: #9b9bb3;
    font-size: 16px;
    padding: 40px 20px;
}

/* Scrollbar styling */
#chat-display::-webkit-scrollbar {
    width: 8px;
}

#chat-display::-webkit-scrollbar-track {
    background: #121218;
}

#chat-display::-webkit-scrollbar-thumb {
    background: #4c3a8f;
    border-radius: 6px;
}

#chat-display::-webkit-scrollbar-thumb:hover {
    background: #6a54c5;
}

/* Message input */
#message-input textarea {
    border-radius: 22px !important;
    border: 1px solid #2a2a3d !important;
    padding: 14px 18px !important;
    font-size: 15px !important;
    background: #16161f !important;
    color: #e6e6f0 !important;
    transition: all 0.25s ease !important;
}

#message-input textarea:focus {
    border-color: #7c5cff !important;
    box-shadow: 0 0 0 2px rgba(124, 92, 255, 0.25) !important;
}

/* Upload button */
.file-upload-btn {
    border-radius: 50% !important;
    width: 52px !important;
    height: 52px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: linear-gradient(135deg, #6c4cff 0%, #9b6dff 100%) !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(108, 76, 255, 0.35) !important;
    transition: all 0.25s ease !important;
}

.file-upload-btn:hover {
    transform: translateY(-2px) scale(1.05) !important;
    box-shadow: 0 6px 20px rgba(108, 76, 255, 0.55) !important;
}

/* Send button */
#send-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    border-radius: 24px !important;
    transition: all 0.3s ease !important;
}

#send-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Clear button */
#clear-btn {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    border-radius: 20px !important;
    transition: all 0.3s ease !important;
}

#clear-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.5) !important;
}

/* Generated image */
#result-image {
    border-radius: 12px;
    border: 1px solid #2a2a3d;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
}

/* Title */
#title-header {
    text-align: center;
    background: linear-gradient(135deg, #7c5cff 0%, #a67cff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.4em;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Subtitle */
#subtitle {
    text-align: center;
    color: #9b9bb3;
    font-size: 1.05em;
    margin-bottom: 30px;
}


"""

# ============================================================================
# Gradio UI
# ============================================================================

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # gr.HTML('<h1 id="title-header">✨ Fashion AI Assistant</h1>')
    gr.HTML('<h1 id="title-header">✨ دستیار هوش مصنوعی بوتیک شما</h1>')
    gr.HTML('<p id="subtitle">عکستون رو با لباس های مختلف به صورت مجازی پرو کنید • درمورد عکستون و استایلتون سوال بپرسید • نتیجه پروتون رو بدون هزینه ببینید!</p>')
    # gr.HTML('<p id="subtitle">Upload your photo and clothing items • Chat naturally • Get instant try-on results</p>')
    
    session_state = gr.State(SessionState())
    
    with gr.Row(elem_id="main-container"):
        
        with gr.Column(scale=2):
            chat_display = gr.HTML(
                # value='<div class="welcome-msg">👋 Welcome! Upload your photo and a clothing item, then ask me to try it on you!</div>',
                value='<div class="welcome-msg">👋 خوش آمدید! عکس و یک لباس خود را آپلود کنید، سپس از من بخواهید آن را روی شما پرو کنم.</div>',
                elem_id="chat-display",
                label="Conversation"
            )
            gr.HTML("<div style='height: 20px;'></div>")
            with gr.Row():
                with gr.Column(scale=9):
                    message_input = gr.Textbox(
                        placeholder="بپرس : «می‌توانم این ژاکت را پرو کنم؟» یا «این لباس را به من نشان بده» یا «چه اکسسوری‌هایی با آن ست می‌شود؟»",
                        show_label=False,
                        lines=1,
                        max_lines=10,
                        elem_id="message-input",
                        autofocus=True
                    )
                
                with gr.Column(scale=1, min_width=70):
                    file_upload = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="📎",
                        elem_classes="file-upload-btn",
                        show_label=False
                    )
                
                with gr.Column(scale=1, min_width=100):
                    send_btn = gr.Button("Send", elem_id="send-btn")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", elem_id="clear-btn", size="sm")
        
        with gr.Column(scale=1):
            result_image = gr.Image(
                label="✨ Try-On Result",
                type="pil",
                elem_id="result-image",
                height=500
            )
            
            # gr.Markdown("""
            # ### 💡 How it works:
            # 1. **Upload** your photo
            # 2. **Upload** a clothing item
            # 3. **Ask** me to try it on
            # 4. **Get** instant results!
            
            # Then ask for accessory suggestions based on your look!
            
            # ---
            
            # **Session ID:** Each browser tab gets a unique session for privacy.
            # """)

            gr.Markdown("""
            ### 💡 چجوری کار میکنه؟:
            1. ✅آپلود کن عکس خودت رو
            2. ✅آپلود کن عکس لباسی که میخوای بپوشی رو هم
            3. ❤️پرو کنم ازم بخواه که برات
            4.  😊عکس شما آماده هست!
            
            یادت نره میتونم درباره عکست نظر بدم یا اگه دنبال اکسسوری هستی بهت پیشنهاد بدم💕✅
            
            ---
            
            **Session ID:** Each browser tab gets a unique session for privacy.
            """)
    
    def handle_submit(message, files, state):
        chat, image, state = process_message(message, files, state)
        return chat, image, state, "", None
    
    send_btn.click(
        fn=handle_submit,
        inputs=[message_input, file_upload, session_state],
        outputs=[chat_display, result_image, session_state, message_input, file_upload]
    )
    
    message_input.submit(
        fn=handle_submit,
        inputs=[message_input, file_upload, session_state],
        outputs=[chat_display, result_image, session_state, message_input, file_upload]
    )
    
    clear_btn.click(
        fn=clear_session,
        inputs=[session_state],
        outputs=[chat_display, result_image, session_state]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

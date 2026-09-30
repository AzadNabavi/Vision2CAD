import anthropic
import base64
from google import genai
from google.genai import types

print("=== Testing Anthropic (Claude) ===")
try:
    client = anthropic.Anthropic(api_key="YOUR_API_KEY")
    
    with open("technical_drawing.jpg", "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": "What is the overall length of the part in this drawing? One number only."}
            ]
        }]
    )
    text_blocks = [c for c in response.content if getattr(c, "type", "") == "text"]
    print(f"Claude claude-opus-5 Response: {text_blocks[0].text if text_blocks else '[no text]'}")
    print("Claude API: OK")
except Exception as e:
    print(f"Claude API: FAILED - {e}")

print()
print("=== Testing Google Gemini ===")
try:
    client_g = genai.Client(api_key="AIzaSyCEhfdP7w-PB3TvdLlXs7koi12AAUbOdYo")
    
    with open("technical_drawing.jpg", "rb") as f:
        img_bytes = f.read()
        
    response = client_g.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            "What is the overall length of the part in this drawing? One number only."
        ]
    )
    print(f"Gemini gemini-2.5-pro Response: {response.text.strip()}")
    print("Gemini API: OK")
except Exception as e:
    print(f"Gemini API: FAILED - {e}")

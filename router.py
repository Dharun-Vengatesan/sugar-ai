import os
import json
import asyncio
import logging
from typing import List, Optional, AsyncIterable
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Primary Model SDK (Google)
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# Fallback Model SDK (Groq)
from groq import AsyncGroq

router = APIRouter()
logger = logging.getLogger("sugar_router")

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# --- Pydantic Models ---

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

# --- Format Converters ---

def convert_to_gemini(messages: List[Message]):
    """Converts OpenAI/Groq format to Gemini format."""
    gemini_history = []
    for msg in messages[:-1]:
        role = "user" if msg.role == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg.content]})
    
    # The last message is the prompt
    last_msg = messages[-1].content
    return gemini_history, last_msg

# --- Core Streaming Logic ---

async def get_gemini_stream(messages: List[Message], temp: float):
    """Attempts to stream from Gemini 1.5 Flash."""
    history, prompt = convert_to_gemini(messages)
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(history=history)
    
    # Using async generation
    response = await chat.send_message_async(
        prompt, 
        stream=True, 
        generation_config=genai.types.GenerationConfig(temperature=temp)
    )
    async for chunk in response:
        if chunk.text:
            yield chunk.text

async def get_groq_stream(messages: List[Message], temp: float):
    """Fallback stream using Groq Llama-3.1-8b."""
    # Groq uses standard OpenAI format
    payload = [{"role": m.role, "content": m.content} for m in messages]
    
    stream = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=payload,
        temperature=temp,
        stream=True,
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

async def sugar_routing_generator(messages: List[Message], temp: float):
    """
    Main Fallback Matrix:
    Tries Gemini -> On (429, Timeout, Auth Error) -> Switches to Groq
    """
    use_fallback = False
    
    try:
        # We attempt to initialize the Gemini stream
        # Note: We must consume the first part of the generator to catch initial connection errors
        gen = get_gemini_stream(messages, temp)
        async for chunk in gen:
            yield f"data: {json.dumps({'content': chunk})}\n\n"
            
    except (google_exceptions.ResourceExhausted, 
            google_exceptions.DeadlineExceeded, 
            google_exceptions.Unauthenticated,
            google_exceptions.ServiceUnavailable,
            Exception) as e:
        
        logger.error(f"Primary Provider (Gemini) Failed: {str(e)}. Routing to Groq...")
        use_fallback = True

    if use_fallback:
        try:
            async for chunk in get_groq_stream(messages, temp):
                yield f"data: {json.dumps({'content': chunk, 'fallback': True})}\n\n"
        except Exception as e:
            logger.critical(f"All providers failed: {str(e)}")
            yield f"data: {json.dumps({'error': 'All service providers are currently unavailable.'})}\n\n"
    
    yield "data: [DONE]\n\n"

# --- API Endpoint ---

@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """
    High-availability chat endpoint for SUGAR.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    return StreamingResponse(
        sugar_routing_generator(request.messages, request.temperature),
        media_type="text/event-stream"
    )

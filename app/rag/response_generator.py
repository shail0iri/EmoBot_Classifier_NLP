import os
from typing import Dict, List
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class ResponseGenerator:
    @staticmethod
    def _normalize_emotion(value):
        if isinstance(value, np.ndarray):
            if value.size == 1:
                value = value.reshape(-1)[0]
            else:
                value = value.tolist()

        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass

        if isinstance(value, (list, tuple)):
            value = value[0] if len(value) == 1 else value

        return str(value).lower().strip()

    def __init__(
        self,
        api_key=None,
        model_name="llama-3.3-70b-versatile",  # Changed to Groq model
        timeout_seconds=20,
        use_llm=True,
    ):
        self.mock_mode = True
        self.use_llm = use_llm and use_llm
        
        if not self.use_llm:
            print("LLM disabled. Using mock responses.")
            return

        # Try Groq API first
        api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.timeout_seconds = timeout_seconds

        if not api_key:
            print("No API key found. Using mock responses.")
            return

        try:
            from groq import Groq
            
            self.client = Groq(api_key=api_key)
            self.model_name = model_name
            self.mock_mode = False
            print(f"✅ Initialized Groq with model: {model_name}")
            return
            
        except ImportError:
            print("⚠️ groq not installed. Run: pip install groq")
            return
        except Exception as e:
            print(f"❌ Failed to initialize Groq: {e}")
            return

    def generate_response(
        self,
        user_query: str,
        emotion: str,
        retrieved_docs: List[str],
        confidence: float,
    ) -> Dict:
        emotion_label = self._normalize_emotion(emotion)
        context = "\n\n".join(retrieved_docs[:3]) if retrieved_docs else "No specific knowledge available."

        if self.mock_mode:
            return self._mock_response(emotion_label)

        prompt = f"""You are an empathetic emotional support assistant.

USER'S EMOTION: {emotion_label}
CONFIDENCE: {confidence*100:.1f}%
USER'S MESSAGE: {user_query}

RELEVANT INFORMATION FROM KNOWLEDGE BASE:
{context}

INSTRUCTIONS:
1. Acknowledge and validate the user's emotion first.
2. Use the relevant information above to provide specific, actionable support.
3. Keep the response warm, concise, and helpful (max 150 words).
4. If the user expresses crisis or self-harm thoughts, include crisis helpline numbers.
5. Do not provide medical advice. Suggest professional help when appropriate.

Your response:"""
        
        try:
            # Groq API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are EmoBot, a compassionate emotional support assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300,
                timeout=self.timeout_seconds,
            )
            
            response_text = response.choices[0].message.content
            
            return {
                "response": response_text,
                "emotion": emotion_label,
                "confidence": confidence,
                "used_context": True,
                "context_sources": len(retrieved_docs),
            }
            
        except Exception as e:
            print(f"LLM error: {e}")
            return self._mock_response(emotion_label)

    def _get_fallback_response(self, emotion_label: str) -> str:
        """Quick fallback if Groq returns empty - ONLY 3 emotions"""
        fallbacks = {
            "anger": "I hear your frustration. Try taking 5 deep breaths before responding.",
            "fear": "Fear is trying to protect you. Let's ground ourselves with 5-4-3-2-1.",
            "joy": "That's beautiful. Hold onto this positive feeling.",
        }
        return fallbacks.get(emotion_label, "Thank you for sharing. Your feelings matter.")

    def _mock_response(self, emotion):
        """Mock responses for ONLY 3 emotions: anger, fear, joy"""
        emotion_label = self._normalize_emotion(emotion)
        responses = {
            "anger": "I can sense your frustration. It often helps to take a brief time-out: step away for 5 minutes, unclench your body, and breathe slowly before responding.",
            "fear": "Fear can feel overwhelming. Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you feel, 3 you hear, 2 you smell, and 1 you taste.",
            "joy": "It is wonderful that you are feeling joyful. Pause for a moment and notice what made this feel meaningful, because savoring positive emotions can build resilience.",
        }
        
        response_text = responses.get(emotion_label, "Thank you for sharing. Your emotions are valid.")
        
        return {
            "response": response_text,
            "emotion": emotion_label,
            "confidence": 0.0,
            "used_context": False,
            "context_sources": 0,
            "mock": True,
        }
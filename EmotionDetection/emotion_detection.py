cat > /home/project/final_project/EmotionDetection/emotion_detection.py << 'EOF'
import requests
import json
from typing import Dict, Any, Optional

_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
_HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

def emotion_detector(text_to_analyze: Optional[str]) -> dict:
    """
    Calls Watson NLP EmotionPredict and returns:
    {
      'anger': float|None, 'disgust': float|None, 'fear': float|None,
      'joy': float|None, 'sadness': float|None, 'dominant_emotion': str|None
    }
    On blank/invalid input (HTTP 400 or exceptions), returns all None.
    """
    def none_result() -> dict:
        return {
            "anger": None, "disgust": None, "fear": None,
            "joy": None, "sadness": None, "dominant_emotion": None
        }

    # Normalize None to empty string so request is consistent
    if text_to_analyze is None:
        text_to_analyze = ""

    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        resp = requests.post(_URL, headers=_HEADERS, json=payload, timeout=15)

        # Required by Task 7: if service returns 400 → all None
        if resp.status_code == 400:
            return none_result()

        resp.raise_for_status()
        data: Dict[str, Any] = json.loads(resp.text)
    except Exception:
        return none_result()

    preds = data.get("emotionPredictions", [])
    if not preds:
        return none_result()

    emotion: Dict[str, float] = preds[0].get("emotion", {})
    scores: Dict[str, float] = {
        "anger": float(emotion.get("anger", 0.0)),
        "disgust": float(emotion.get("disgust", 0.0)),
        "fear": float(emotion.get("fear", 0.0)),
        "joy": float(emotion.get("joy", 0.0)),
        "sadness": float(emotion.get("sadness", 0.0)),
    }

    # Dominant only if we actually have numeric scores; otherwise None
    scores["dominant_emotion"] = max(scores, key=scores.get) if any(scores.values()) else None
    return scores
EOF


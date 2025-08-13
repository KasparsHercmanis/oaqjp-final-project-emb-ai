"""Flask web server for the Emotion Detection application."""

from flask import Flask, request
from EmotionDetection import emotion_detector

app = Flask(__name__)

@app.route("/emotionDetector", methods=["POST"])
def detect_emotion():
    """
    Handle POST requests to the /emotionDetector endpoint.

    Returns:
        tuple[str, int]: A response message and HTTP status code.
    """
    # Extract text from JSON (preferred) or HTML form fallback
    input_text = ""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        input_text = (data.get("text") or "").strip()
    else:
        input_text = (request.form.get("textToAnalyze") or "").strip()

    # Blank input handling (Task 7)
    if not input_text:
        return "Invalid text! Please try again!", 400

    # Call the emotion detection function
    scores = emotion_detector(input_text)

    # If API mapped input to an invalid case, return friendly error
    if scores.get("dominant_emotion") is None:
        return "Invalid text! Please try again!", 400

    # Build the required formatted message
    message = (
        "For the given statement, the system response is "
        f"'anger': {scores['anger']}, 'disgust': {scores['disgust']}, "
        f"'fear': {scores['fear']}, 'joy': {scores['joy']} and "
        f"'sadness': {scores['sadness']}. The dominant emotion is "
        f"{scores['dominant_emotion']}."
    )

    return message, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

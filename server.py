from flask import Flask, request, render_template
from EmotionDetection import emotion_detector

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# Must be exactly /emotionDetector
@app.route("/emotionDetector", methods=["GET", "POST"])
def emotion_detector_route():
    """
    Accept text via JSON/form/query, call emotion_detector,
    return the required formatted string; if blank/invalid -> friendly error.
    """
    text_to_analyze = None

    # JSON: {"text": "..."}
    if request.is_json:
        data = request.get_json(silent=True) or {}
        text_to_analyze = data.get("text")

    # Fallbacks: form/query
    if not text_to_analyze:
        text_to_analyze = (
            request.form.get("textToAnalyze")
            or request.args.get("textToAnalyze")
            or request.form.get("text")
            or request.args.get("text")
        )

    # Always call our function (it maps 400/invalid to all-None dict)
    scores = emotion_detector(text_to_analyze)

    # Error handling per Task 7
    if scores.get("dominant_emotion") is None:
        return "Invalid text! Please try again!", 400

    msg = (
        f"For the given statement, the system response is "
        f"'anger': {scores['anger']}, 'disgust': {scores['disgust']}, "
        f"'fear': {scores['fear']}, 'joy': {scores['joy']} and "
        f"'sadness': {scores['sadness']}. The dominant emotion is "
        f"{scores['dominant_emotion']}."
    )
    return msg, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

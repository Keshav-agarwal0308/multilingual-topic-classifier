from langdetect import detect, LangDetectException
TOPIC_EMOJIS = {
    "geography": "🌍",
    "science/technology": "🔬",
    "entertainment": "🎬",
    "politics": "🏛️",
    "health": "🏥",
    "travel": "✈️",
    "sports": "⚽"
}
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ja": "Japanese",
    "zh-cn": "Chinese",
    "ar": "Arabic"
}
def detect_language(text):
    try:
        code = detect(text)
        return LANGUAGE_NAMES.get(code, code)
    except LangDetectException:
        return "Unknown"
def get_topic_emoji(topic):
    return TOPIC_EMOJIS.get(topic.lower(), "📌")
def format_prediction(topic, confidence):
    emoji = get_topic_emoji(topic)
    return {
        "topic": f"{emoji} {topic.title()}",
        "confidence": f"{confidence:.2f}%"
    }
def validate_input(text):
    if not text:
        return False
    if not text.strip():
        return False
    return True

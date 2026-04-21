import gradio as gr
from transformers import pipeline
from langdetect import detect, LangDetectException

# Load model from HuggingFace
classifier = pipeline(
    "text-classification",
    model="Keshav0308/multilingual-topic-classifier"
)

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
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh-cn": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "bn": "Bengali", "ur": "Urdu", "tr": "Turkish", "pl": "Polish",
    "nl": "Dutch", "sv": "Swedish", "fi": "Finnish", "da": "Danish",
    "uk": "Ukrainian", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
    "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
    "fa": "Persian", "he": "Hebrew", "pa": "Punjabi", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "si": "Sinhala", "ne": "Nepali", "am": "Amharic",
    "sw": "Swahili", "yo": "Yoruba", "ig": "Igbo", "ha": "Hausa",
    "zu": "Zulu", "af": "Afrikaans", "sq": "Albanian", "hy": "Armenian",
    "az": "Azerbaijani", "eu": "Basque", "be": "Belarusian", "bs": "Bosnian",
    "bg": "Bulgarian", "ca": "Catalan", "hr": "Croatian", "et": "Estonian",
    "gl": "Galician", "ka": "Georgian", "el": "Greek", "is": "Icelandic",
    "lv": "Latvian", "lt": "Lithuanian", "mk": "Macedonian", "mt": "Maltese",
    "sr": "Serbian", "sk": "Slovak", "sl": "Slovenian", "cy": "Welsh",
}

def detect_language(text):
    try:
        code = detect(text)
        return LANGUAGE_NAMES.get(code, f"Unknown ({code})")
    except LangDetectException:
        return "Could not detect"

def classify_topic(text):
    if not text or not text.strip():
        return "", "", ""
    
    result = classifier(text)[0]
    topic = result["label"]
    confidence = result["score"] * 100
    language = detect_language(text)
    
    emoji = TOPIC_EMOJIS.get(topic, "📌")
    topic_display = f"{emoji} {topic.upper()}"
    confidence_display = f"{confidence:.2f}%"
    language_display = f"🌐 {language}"
    
    return topic_display, confidence_display, language_display

# Example inputs
examples = [
    ["The patient was diagnosed with pneumonia and prescribed antibiotics."],
    ["El equipo ganó el campeonato mundial de fútbol."],
    ["Le parlement a voté une nouvelle loi sur l'environnement."],
    ["scientists discovered a new exoplanet orbiting a distant star."],
    ["ਕ੍ਰਿਕੇਟ ਟੀਮ ਨੇ ਵਿਸ਼ਵ ਕੱਪ ਜਿੱਤਿਆ।"],
    ["東京オリンピックで日本が金メダルを獲得した。"],
    ["Der Bundestag hat ein neues Klimaschutzgesetz verabschiedet."],
]

# Build UI
with gr.Blocks(theme=gr.themes.Soft(), title="Multilingual Topic Classifier") as demo:
    gr.Markdown("""
    # 🌍 Multilingual Topic Classifier
    ### Classify text into topics across 205 languages
    Built with `xlm-roberta-base` fine-tuned on the SIB-200 dataset.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter text in any language",
                placeholder="Type or paste text here...",
                lines=4
            )
            submit_btn = gr.Button("🔍 Classify", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            topic_output = gr.Textbox(label="📌 Topic", interactive=False)
            confidence_output = gr.Textbox(label="📊 Confidence", interactive=False)
            language_output = gr.Textbox(label="🌐 Detected Language", interactive=False)
    
    gr.Examples(
        examples=examples,
        inputs=text_input,
        label="Try these examples"
    )
    
    submit_btn.click(
        fn=classify_topic,
        inputs=text_input,
        outputs=[topic_output, confidence_output, language_output]
    )
    
    text_input.submit(
        fn=classify_topic,
        inputs=text_input,
        outputs=[topic_output, confidence_output, language_output]
    )

demo.launch()
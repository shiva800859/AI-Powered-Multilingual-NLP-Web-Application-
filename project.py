import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from textblob import TextBlob
from nltk.corpus import wordnet
import nltk

# --- 1. DOWNLOAD HELPER DATA ---
# This ensures the dictionary and spelling tools work on your laptop
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('brown')

print("⏳ Loading AI Models... (This uses your laptop's memory)")

# --- 2. LOAD AI MODELS ---

# A. Translation Model
# We use the CPU by default so it doesn't crash if you don't have an NVIDIA GPU
trans_model_name = "facebook/nllb-200-distilled-600M"
trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)
translator_pipeline = pipeline("translation", model=trans_model, tokenizer=trans_tokenizer)

# B. English Sentiment Model
eng_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

print("✅ Models Loaded! Launching App...")

# --- 3. LOGIC FUNCTIONS ---

lang_codes = {
    "Telugu": "tel_Telu", "Hindi": "hin_Deva", "Tamil": "tam_Taml", "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym", "French": "fra_Latn", "Spanish": "spa_Latn", "German": "deu_Latn",
    "Italian": "ita_Latn", "Japanese": "jpn_Jpan", "Chinese (Simplified)": "zho_Hans", "Russian": "rus_Cyrl",
    "Arabic": "ara_Arab", "Korean": "kor_Hang", "Portuguese": "por_Latn", "Dutch": "nld_Latn",
    "Turkish": "tur_Latn", "Vietnamese": "vie_Latn", "Thai": "tha_Thai", "Indonesian": "ind_Latn",
    "Polish": "pol_Latn", "Ukrainian": "ukr_Cyrl", "Greek": "ell_Grek", "Bengali": "ben_Beng"
}

def translate_text(text, target_lang):
    if not text: return "Please enter text."
    target_code = lang_codes.get(target_lang, "tel_Telu")
    translator = pipeline("translation", model=trans_model, tokenizer=trans_tokenizer, src_lang="eng_Latn", tgt_lang=target_code)
    # CPU translation might be slower, so we limit length slightly for speed
    result = translator(text, max_length=200)
    return result[0]['translation_text']

def correct_spelling(text):
    if not text: return "Please enter text."
    blob = TextBlob(text)
    return str(blob.correct())

def get_definition(word):
    if not word: return "Please enter a word."
    syns = wordnet.synsets(word)
    if not syns: return "Definition not found."
    return f"Definition of '{word}':\n{syns[0].definition()}"

def analyze_eng_sentiment(text):
    if not text: return "Please enter text."
    result = eng_sentiment(text)
    label = result[0]['label']
    score = round(result[0]['score'], 2)
    return f"Sentiment: {label} (Confidence: {score})"

def analyze_tel_sentiment(text):
    if not text: return "Please enter text."
    
    # Step 1: Translate Telugu -> English
    tel_to_eng = pipeline("translation", model=trans_model, tokenizer=trans_tokenizer, src_lang="tel_Telu", tgt_lang="eng_Latn")
    translated_data = tel_to_eng(text, max_length=200)
    english_text = translated_data[0]['translation_text']
    
    # Step 2: Analyze English Sentiment
    result = eng_sentiment(english_text)
    label = result[0]['label']
    score = round(result[0]['score'], 2)
    return f"Original: {text}\nTranslated: {english_text}\nSentiment: {label} (Confidence: {score})"

# --- 4. BUILD APP UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 All-in-One NLP Toolkit (Local Version)")
    
    with gr.Tab("Translation"):
        gr.Markdown("Translate English to 24 Languages")
        t_in = gr.Textbox(label="English Text")
        t_lang = gr.Dropdown(list(lang_codes.keys()), label="Target Language", value="Telugu")
        t_btn = gr.Button("Translate")
        t_out = gr.Textbox(label="Result")
        t_btn.click(translate_text, inputs=[t_in, t_lang], outputs=t_out)

    with gr.Tab("Spelling"):
        gr.Markdown("Correct English Spelling")
        s_in = gr.Textbox(label="Text with errors")
        s_btn = gr.Button("Correct")
        s_out = gr.Textbox(label="Result")
        s_btn.click(correct_spelling, inputs=s_in, outputs=s_out)

    with gr.Tab("Definition"):
        gr.Markdown("English Word Definitions")
        d_in = gr.Textbox(label="Word")
        d_btn = gr.Button("Define")
        d_out = gr.Textbox(label="Definition")
        d_btn.click(get_definition, inputs=d_in, outputs=d_out)

    with gr.Tab("English Sentiment"):
        gr.Markdown("English Sentiment Analysis")
        es_in = gr.Textbox(label="English Text")
        es_btn = gr.Button("Analyze")
        es_out = gr.Textbox(label="Result")
        es_btn.click(analyze_eng_sentiment, inputs=es_in, outputs=es_out)

    with gr.Tab("Telugu Sentiment"):
        gr.Markdown("Telugu Sentiment Analysis")
        ts_in = gr.Textbox(label="Telugu Text", placeholder="ఈ సినిమా చాలా బాగుంది")
        ts_btn = gr.Button("Analyze")
        ts_out = gr.Textbox(label="Result")
        ts_btn.click(analyze_tel_sentiment, inputs=ts_in, outputs=ts_out)

# Launch the app
demo.launch()
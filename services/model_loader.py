# utils/model_loader.py

import os
import whisper
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
from speechbrain.pretrained import EncoderClassifier


def load_whisper_model(model_name="medium", save_dir="pretrained_models/whisper"):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}.pt")

    if os.path.exists(model_path):
        print(f"⚡ Whisper: Загружаем модель из локального файла {model_path}")
        model = whisper.load_model(model_path)
    else:
        print(f"🔽 Whisper: Скачиваем модель '{model_name}' в {save_dir}...")
        model = whisper.load_model(model_name, download_root=save_dir)
        print("✅ Whisper модель загружена.")

    return model


def load_bert_tokenizer(model_name="bert-base-multilingual-cased", save_dir="pretrained_models/bert"):
    print("🔐 Загружаем переменные окружения...")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variables.")

    login(hf_token)
    print(f"🔽 Скачиваем BERT токенизатор в {save_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_dir)
    print("✅ BERT токенизатор готов.")
    return tokenizer


def load_speechbrain_encoder(save_dir="pretrained_models/speechbrain"):
    print(f"🔽 Скачиваем SpeechBrain модель в {save_dir}...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=save_dir
    )
    print("✅ SpeechBrain модель готова.")
    return model

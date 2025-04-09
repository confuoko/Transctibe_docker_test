from services.model_loader import (
    load_whisper_model,
    load_bert_tokenizer,
    load_speechbrain_encoder
)

if __name__ == "__main__":
    print("📦 Предварительная загрузка всех моделей...")
    whisper_model = load_whisper_model()
    tokenizer = load_bert_tokenizer()
    speechbrain_encoder = load_speechbrain_encoder()
    print("🎉 Все модели успешно загружены и сохранены.")

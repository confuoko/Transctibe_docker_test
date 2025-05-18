# Документация по сервису транскрибации

## Описание

Этот проект представляет собой **сервис транскрибации**, который принимает аудиофайл, выполняет его обработку с использованием модели Whisper для транскрибации и модели для диаризации (определения спикеров). Система также поддерживает интеграцию с сервисом кластеризации для дальнейшей обработки транскрибированного текста.

Процесс обработки включает:
1. Загрузка аудиофайла из облачного хранилища (S3).
2. Выполнение транскрибации аудио с помощью модели Whisper.
3. Диаризация (разделение аудио на сегменты, принадлежащие разным спикерам).
4. Объединение результатов транскрибации и диаризации.
5. Сохранение результатов в базе данных и в облачном хранилище.
6. Отправка результатов в сервис кластеризации.

## Функции

- **Загрузка аудиофайла**: Аудиофайл загружается из облачного хранилища S3.
- **Транскрибация**: Аудиофайл транскрибируется с использованием модели Whisper.
- **Диаризация**: Разделение аудио на сегменты, принадлежащие разным спикерам, с использованием модели Silero VAD и кластеризации с помощью KMeans.
- **Объединение результатов**: Объединение транскрибированных данных и результатов диаризации в конечный текст.
- **Сохранение в S3**: Текст сохраняется в облачное хранилище S3.
- **Обновление записи в базе данных**: Результаты транскрибации сохраняются в базе данных для дальнейшей обработки.

## Установка

### Требования

- Python 3.8+
- Библиотеки:
  - `boto3` (для работы с S3)
  - `torch` (для работы с моделью Whisper)
  - `requests` (для отправки запросов)
  - `dotenv` (для работы с переменными окружения)
  - `sklearn` (для кластеризации)
  - `psycopg2` (для работы с базой данных)
  - `pydub` (для работы с аудиофайлами)
  - `silero_vad` (для диаризации)

### Шаги установки

1. Клонируйте репозиторий:
   ```bash
   git clone <url-репозитория>
   cd <папка-проекта>

Как работает транскрибация
1. Загрузка файла из S3
Аудиофайл загружается из облачного хранилища S3 в локальную временную папку.

python
Копировать
file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
temp_file_path = os.path.join(temp_dir, file_key)

with open(temp_file_path, 'wb') as temp_file:
    temp_file.write(file_obj['Body'].read())
2. Транскрибация аудио
Файл транскрибируется с использованием модели Whisper.

python
Копировать
def transcribe(file_path):
    print("TRANSCRIBING...")
    return whisper_model.transcribe(file_path)
3. Диаризация (разделение на спикеров)
Используется модель Silero VAD для получения сегментов речи и кластеризация с помощью KMeans для определения спикеров.

python
Копировать
def diarize(file_path):
    print("DIARIZING...")
    vad_model = load_silero_vad()
    audio = read_audio(file_path)
    speech_timestamps = get_speech_timestamps(audio, vad_model, return_seconds=True)

    embeddings = []
    for segment in speech_timestamps:
        start = int(segment['start'] * 16000)
        end = int(segment['end'] * 16000)
        segment_wav = audio[start:end].unsqueeze(0)
        embedding = speechbrain_model.encode_batch(segment_wav)
        embeddings.append(embedding.squeeze().numpy())

    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(embeddings)
    return speech_timestamps, labels
4. Объединение результатов
Результаты транскрибации и диаризации объединяются для создания финального текста, где каждому сегменту аудио присваивается спикер.

python
Копировать
def unite_results(transcribed_result, diarized_result, labels):
    print("UNITING RESULTS..")
    diarization_result = []
    base_string_res = []
    for i, segment in enumerate(diarized_result):
        speaker = f"Speaker_{labels[i] + 1}"
        diarization_result.append({
            "start": segment['start'],
            "end": segment['end'],
            "speaker": speaker
        })
    silero_vad_speakers = []
    for segment in transcribed_result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        max_overlap = 0
        best_speaker = None

        for diarization_segment in diarization_result:
            overlap_start = max(start, diarization_segment["start"])
            overlap_end = min(end, diarization_segment["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diarization_segment["speaker"]

        if best_speaker is None:
            for diarization_segment in diarization_result:
                if diarization_segment["end"] >= start or diarization_segment["start"] <= end:
                    best_speaker = diarization_segment["speaker"]
                    break

        speaker = best_speaker if best_speaker else "Unknown"
        silero_vad_speakers.append((speaker, text))
        base_string_res.append(text)
    for view_res in silero_vad_speakers:
        print(f'Speaker: {view_res[0]} - {view_res[1]}')

    # Формируем строку с текстами, разделенными по спикерам
    result_str = ""
    for speaker, text in silero_vad_speakers:
        result_str += f"{speaker}: {text}"  # Добавляем информацию по каждому спикеру

    return result_str.strip()  # Убираем лишний перенос в конце
5. Сохранение результатов
Текст сохраняется в файл и загружается обратно в S3.

python
Копировать
with open(text_file_path, "w", encoding="utf-8") as f:
    f.write(answer)
    f.write(f"\n\n⏱️ Файл был обработан за {duration} секунд.")
    f.write(f"\n\n⏱️ Доступность CUDA: {torch_variable}.")
6. Отправка запроса в сервис кластеризации
После транскрибации текст отправляется в сервис кластеризации для дальнейшей обработки.

python
Копировать
data = {
    "input": {
        "file_key": text_filename,
        "item_id": item_id
    }
}
response = requests.post(f'https://api.runpod.ai/v2/{CLASTERIZATOR_QUEUE}/run', headers=headers, json=data)


Лицензия
Этот проект лицензирован под лицензией MIT — смотрите файл LICENSE для подробностей.
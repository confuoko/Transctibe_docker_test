import argparse
import os
import boto3
import time

import requests
import torch
from services.transcribe_service import processFile, update_record
import runpod
from dotenv import load_dotenv


def handler(event):
    torch_variable = str(torch.cuda.is_available())
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = "whisper-audiotest"

    print(torch_variable)
    # Получает от сервиса шумоподавления очищенное аудио и id

    file_key = event['input']['file_key']
    item_id = event['input']['item_id']

    #file_key = 'interview1.wav'
    #item_id = 1

    print(f"Получен запрос на транскрибацию файла: {file_key}")
    print(f"ID элемента: {item_id}")

    load_dotenv()
    s3 = boto3.client(
        's3',
        endpoint_url="http://storage.yandexcloud.net",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)

    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file_key)

    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(file_obj['Body'].read())

    print(f'📁 Файл сохранён во временную папку: {temp_file_path}')

    # === Засекаем время ===
    print("🧠 Начинаю транскрибацию...")
    start_time = time.time()
    answer = processFile(temp_file_path)
    duration = round(time.time() - start_time, 2)  # Время в секундах с округлением
    print(f"✅ Готово за {duration} секунд.")

    # === Генерируем имя текстового файла ===
    text_filename = os.path.splitext(file_key)[0] + ".txt"
    text_file_path = os.path.join(temp_dir, text_filename)

    # === Сохраняем текст во временный .txt файл ===
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(answer)
        f.write(f"\n\n⏱️ Файл был обработан за {duration} секунд.")
        f.write(f"\n\n⏱️ Доступность CUDA: {torch_variable}.")

    print(f"📝 Ответ сохранён в файл: {text_file_path}")

    # === Загружаем .txt обратно в S3 ===
    s3.upload_file(
        Filename=text_file_path,
        Bucket=bucket_name,
        Key=text_filename
    )
    print(f"☁️ Файл {text_filename} загружен в S3 (в корень бакета)")

    # === Удаляем временные файлы ===
    os.remove(text_file_path)
    os.remove(temp_file_path)
    print("🧹 Временные файлы удалены.")

    # Сохраняем имя текстового файла в БД
    # пушим данные в БД
    update_record(item_id, text_filename)

    # Отправляем запрос в сервис Кластеризации: item_id, text_filename
    # Переменные для сервиса транскрибации
    RUNPOD_AUTH_TOKEN = os.getenv("RUNPOD_AUTH_TOKEN")
    CLASTERIZATOR_QUEUE = os.getenv("CLASTERIZATOR_QUEUE")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': RUNPOD_AUTH_TOKEN
    }

    data = {
        "input": {
            "file_key": text_filename,
            "item_id": item_id
        }
    }
    print(f"Направляю файл: {text_filename} и ID записи: {item_id}")
    response = requests.post(f'https://api.runpod.ai/v2/{CLASTERIZATOR_QUEUE}/run', headers=headers, json=data)

    print(f"Статус ответа: {response.status_code}")
    print(f"Тело ответа: {response.text}")
    print(f"Отправляем запрос к сервису Кластеризации. Ответ: {response.json()}")


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
    #handler(None)


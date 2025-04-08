import argparse
import os
import boto3
import time
from dotenv import load_dotenv
from services.transcribe_service import processFile

def main():
    #load_dotenv()
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = "whisper-audiotest"

    parser = argparse.ArgumentParser(description="Transcribe file from S3")
    parser.add_argument("-file", required=True, help="Название файла (ключ в S3)")
    args = parser.parse_args()
    file_key = args.file

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

if __name__ == "__main__":
    main()

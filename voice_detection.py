import speech_recognition as sr

# Tạo đối tượng Recognizer
recognizer = sr.Recognizer()

# Đường dẫn tới file âm thanh
audio_file = "path_to_your_audio_file.wav"

# Mở file âm thanh và phân tích
with sr.AudioFile(audio_file) as source:
    # Ghi nhận dữ liệu âm thanh từ file
    audio_data = recognizer.record(source)
    
    # Nhận diện giọng nói và chuyển sang text
    try:
        text = recognizer.recognize_google(audio_data, language='vi-VN')
        print("Text được nhận diện:", text)
    except sr.UnknownValueError:
        print("Không thể nhận diện được giọng nói")
    except sr.RequestError as e:
        print(f"Không thể yêu cầu kết quả từ Google API; {e}")
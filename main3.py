import cv2
from ultralytics import YOLO

# 1. Загружаем модель (убедитесь, что путь к .pt файлу верный)
model = YOLO("models/yolo26x.pt")

# 2. Указываем путь к видеофайлу
video_path = "C:/Users/chelowek/Videos/2026-03-04 17-21-18.mkv"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # 3. Запускаем трекинг (BoT-SORT с фильтром Калмана)
        # persist=True обязателен для сохранения ID между кадрами
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

        # 4. Визуализируем результаты (отрисовываем рамки и ID)
        annotated_frame = results[0].plot()

        # 5. Показываем результат
        cv2.imshow("YOLO Video Tracking", annotated_frame)

        # Выход по нажатию клавиши 'q' или если окно закрыто
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Конец видео
        break

cap.release()
cv2.destroyAllWindows()
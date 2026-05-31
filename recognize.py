import cv2
import numpy as np

# Load model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("models/trainer.yml")

# Load label map
label_map = np.load(
    "models/label_map.npy",
    allow_pickle=True
).item()

# Face detector
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

kamera = cv2.VideoCapture(0)

while True:

    ret, frame = kamera.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    wajah = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x,y,w,h) in wajah:

        face = gray[
            y:y+h,
            x:x+w
        ]

        label, confidence = model.predict(face)

        nama = label_map.get(
            label,
            "Unknown"
        )

        if confidence < 100:

            text = f"{nama} ({confidence:.0f})"

        else:

            text = "Unknown"

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow(
        "Absensi AI",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

kamera.release()
cv2.destroyAllWindows()
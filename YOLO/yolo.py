import cv2
import depthai as dai
from ultralytics import YOLO


def main():
    model_path = "networkWeights.pt"

    print(f"Wczytywanie modelu: {model_path}...")
    model = YOLO(model_path)

    print("\n--- KONFIGURACJA KAMERY OAK-D (API V3) ---")

    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.Camera).build()

    cam_out = cam_rgb.requestOutput((640, 640), type=dai.ImgFrame.Type.BGR888p)

    q_rgb = cam_out.createOutputQueue(maxSize=4, blocking=False)

    print("\n--- URUCHAMIANIE KAMERY ---")
    print("Aby wyłączyć kamerę i zakończyć skrypt, kliknij na okno z obrazem i wciśnij klawisz 'q' na klawiaturze.")

    pipeline.start()

    while pipeline.isRunning():

        in_rgb = q_rgb.get()
        frame = in_rgb.getCvFrame()

        results = model.predict(
            source=frame,  # Przekazujemy macierz z obrazem
            conf=0.5,  # Próg pewności
            device=0,  # Zrzucenie obliczeń na kartę RTX (ID=0)
            show=False,  # OpenCV zajmie się wyświetlaniem
            stream=False  # Musi być False dla pojedynczych klatek z tablic
        )

        # Pobranie obrazu z naniesionymi ramkami predykcji
        annotated_frame = results[0].plot()

        # Wyświetlenie obrazu za pomocą OpenCV
        cv2.imshow("OAK-D S2 - YOLO Live", annotated_frame)

        # Oczekiwanie na klawisz 'q' w celu wyłączenia
        if cv2.waitKey(1) == ord('q'):
            break

    # Sprzątanie
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
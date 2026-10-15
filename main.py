import os
import cv2 as cv
from sources.eye_detector import EyeDetector


def run_system_on_media(source_path=None):
    detector = EyeDetector()

    if source_path is not None and os.path.exists(source_path):
        if source_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame = cv.imread(source_path)
            if frame is None:
                print(f"Failed to load image from {source_path}")
                return
            frame, eye_states = detector.process_frame(frame)
            cv.imshow("Drowsy-Vision Image Test", frame)
            cv.waitKey(0)
            cv.destroyAllWindows()
            return
        else:
            capture = cv.VideoCapture(source_path)
    else:
        capture = cv.VideoCapture(0)

    closed_frame_count = 0
    alert_threshold = 15

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Video ended or source not accessible.")
            break

        frame, eye_states = detector.process_frame(frame)

        if len(eye_states) > 0 and all(state == 0 for state in eye_states):
            closed_frame_count += 1
        else:
            closed_frame_count = 0

        if closed_frame_count >= alert_threshold:
            cv.putText(
                frame,
                "DROWSINESS ALERT!",
                (50, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        cv.imshow("Drowsy-Vision", frame)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    run_system_on_media("test_video.mp4")
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

    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    window_name = "Drowsy-Vision"
    cv.namedWindow(window_name)

    is_video_file = total_frames > 0
    paused = False
    need_reprocess = True
    current_frame = None
    display_frame = None
    closed_frame_count = 0
    alert_threshold = 15

    def on_trackbar(pos):
        nonlocal need_reprocess, current_frame
        current_pos = int(capture.get(cv.CAP_PROP_POS_FRAMES))
        if abs(pos - current_pos) > 1:
            capture.set(cv.CAP_PROP_POS_FRAMES, pos)
            ret, img = capture.read()
            if ret:
                current_frame = img
                need_reprocess = True

    if is_video_file:
        cv.createTrackbar("Frame", window_name, 0, max(1, total_frames - 1), on_trackbar)

    while True:
        if not paused:
            ret, img = capture.read()
            if not ret:
                print("Video ended.")
                paused = True
            else:
                current_frame = img
                need_reprocess = True
                if is_video_file:
                    pos = int(capture.get(cv.CAP_PROP_POS_FRAMES))
                    cv.setTrackbarPos("Frame", window_name, pos)

        if current_frame is not None and need_reprocess:
            display_frame = current_frame.copy()
            display_frame, eye_states = detector.process_frame(display_frame)

            if len(eye_states) > 0 and all(state == 0 for state in eye_states):
                if not paused:
                    closed_frame_count += 1
            else:
                if not paused:
                    closed_frame_count = 0

            if closed_frame_count >= alert_threshold:
                cv.putText(
                    display_frame,
                    "DROWSINESS ALERT!",
                    (50, 60),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )

            status_text = "PAUSED (Space: Play | A/D: Seek | S: Save)" if paused else "PLAYING (Space: Pause)"
            cv.putText(
                display_frame,
                status_text,
                (10, display_frame.shape[0] - 15),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )
            need_reprocess = False

        if display_frame is not None:
            cv.imshow(window_name, display_frame)

        key = cv.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            need_reprocess = True
        elif key == ord('d') and is_video_file:
            cur = int(capture.get(cv.CAP_PROP_POS_FRAMES))
            step = 1 if paused else 15
            target = min(total_frames - 1, cur + step - (0 if not paused else 1))
            capture.set(cv.CAP_PROP_POS_FRAMES, target)
            ret, img = capture.read()
            if ret:
                current_frame = img
                need_reprocess = True
                cv.setTrackbarPos("Frame", window_name, int(capture.get(cv.CAP_PROP_POS_FRAMES)))
        elif key == ord('a') and is_video_file:
            cur = int(capture.get(cv.CAP_PROP_POS_FRAMES))
            step = 5 if paused else 15
            target = max(0, cur - step - 1)
            capture.set(cv.CAP_PROP_POS_FRAMES, target)
            ret, img = capture.read()
            if ret:
                current_frame = img
                need_reprocess = True
                cv.setTrackbarPos("Frame", window_name, int(capture.get(cv.CAP_PROP_POS_FRAMES)))
        elif key == ord('s') and display_frame is not None:
            cur = int(capture.get(cv.CAP_PROP_POS_FRAMES))
            save_name = f"frame_{cur}.png"
            cv.imwrite(save_name, display_frame)
            print(f"Saved screenshot: {save_name}")

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    run_system_on_media("test_video.mp4")
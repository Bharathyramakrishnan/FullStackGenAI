import cv2
import tempfile
import os


def save_uploaded_video(uploaded_file):

    suffix=os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=suffix

    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        return tmp.name


def extract_frame(video_path):

    cap=cv2.VideoCapture(
        video_path
    )

    success,frame=cap.read()

    if not success:

        return None

    frame_path="frame.jpg"

    cv2.imwrite(

        frame_path,

        frame

    )

    cap.release()

    return frame_path


def video_information(video_path):

    cap=cv2.VideoCapture(
        video_path
    )

    fps=cap.get(
        cv2.CAP_PROP_FPS
    )

    frames=cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    duration=frames/fps

    cap.release()

    return {

        "fps":round(fps,2),

        "duration":round(duration,2)

    }
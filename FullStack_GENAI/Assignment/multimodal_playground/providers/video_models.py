import cv2

def extract_frame(video):

    cap=cv2.VideoCapture(video)

    success,frame=cap.read()

    cv2.imwrite(
        "frame.jpg",
        frame
    )

    return "frame.jpg"
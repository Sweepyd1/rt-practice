import cv2
from ultralytics import YOLO


def avi_solution():
    model = YOLO("yolo11n-pose.pt")
    video_path = "./video2.mp4"
    results = model.track(source="video2.mp4", save=True, project='./result')


def mp4_solution():
    model = YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture("video2.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
 
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mp4_solution()
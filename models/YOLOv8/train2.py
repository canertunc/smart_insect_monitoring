from ultralytics import YOLO

def main():

    model = YOLO('yolov8n.pt')

    model.train(
        data=r"C:\Users\akcah\Desktop\DRAW\ortak.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        name="insect_detector",
        workers=4,
        device=0,
        
    )

if __name__ == "__main__":
    main()

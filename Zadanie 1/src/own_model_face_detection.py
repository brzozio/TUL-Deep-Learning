import torch
import cv2
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import os

SRC_PATH:    str = os.path.dirname(os.path.abspath(__file__))
CSV_PATH:    str = os.path.dirname(SRC_PATH) + '\\csv'
JSON_PATH:   str = os.path.dirname(SRC_PATH) + '\\json'
MODELS_PATH: str = os.path.dirname(SRC_PATH) + '\\models'


def load_model(model_path, device):
    # Load the saved model state dictionary
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    
    # Load the model weights
    if os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # Set model to evaluation mode
        
    return model

def detect_faces(model, frame, device):
    # Convert the frame to a PIL Image
    img = Image.fromarray(frame)
    
    # Apply the same transformation used during training
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((300, 300)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Get predictions from the model
    with torch.no_grad():
        prediction = model(img_tensor)
    
    # Extract boxes from predictions
    boxes = prediction[0]['boxes'].cpu().numpy().astype(int)
    return boxes

def draw_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    return frame

if __name__ == '__main__':
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Path to the saved model
    model_path = MODELS_PATH + "\\faster_rcnn_widerface.pth"
    
    # Load the trained model
    model = load_model(model_path, device)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    skip = 5
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process the frame every 'skip' frames
        if i % skip == 0:
            boxes = detect_faces(model, frame, device)
        
        # Draw the bounding boxes on the frame
        frame = draw_boxes(frame, boxes)

        # Show the frame with face detections
        cv2.imshow('Face Detection', frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        i += 1

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

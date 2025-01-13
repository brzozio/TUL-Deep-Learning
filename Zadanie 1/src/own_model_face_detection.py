import torch
import cv2
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import os

SRC_PATH: str = os.path.dirname(os.path.abspath(__file__))
CSV_PATH: str = os.path.dirname(SRC_PATH) + '\\csv'
JSON_PATH: str = os.path.dirname(SRC_PATH) + '\\json'
MODELS_PATH: str = os.path.dirname(SRC_PATH) + '\\models'


def load_model(model_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    
    if os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    else:
        print(f"Error: Model checkpoint not found at {model_path}")
        exit()
        
    return model

def detect_faces(model, frame, device, confidence_threshold=0.5, iou_threshold=0.5):
    img = Image.fromarray(frame)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((800, 800)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)
    
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    
    high_conf_indices = scores > confidence_threshold
    boxes = boxes[high_conf_indices]
    scores = scores[high_conf_indices]

    if len(boxes) > 0:
        keep_indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)
        boxes = boxes[keep_indices.numpy()]
        scores = scores[keep_indices.numpy()]
    
    return boxes.astype(int), scores

def draw_boxes(frame, boxes, scores=None, threshold=0.5):
    original_height, original_width = frame.shape[:2]
    
    scale_x = original_width / 800
    scale_y = original_height / 800
    
    if scores is not None and threshold is not None:
        filtered_indices = scores >= threshold
        boxes = boxes[filtered_indices]
        scores = scores[filtered_indices]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        
        if scores is not None:
            score = scores[i]
            label = f"{score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return frame


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = MODELS_PATH + "\\faster_rcnn_widerface.pth"
    
    model = load_model(model_path, device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    skip = 5
    i = 0
    boxes = []
    scores = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        original_frame = frame.copy()

        if i % skip == 0:
            boxes, scores = detect_faces(model, frame, device)
        
        frame = draw_boxes(original_frame, boxes, scores)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        i += 1

    cap.release()
    cv2.destroyAllWindows()
# streamapp/camera.py
import cv2
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import json


def frame_to_PIL(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


model = models.resnet34(pretrained=False)

state_dict = torch.load("streamapp/resnet34.pth")
model.load_state_dict(state_dict)
model.eval()
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

with open("streamapp/imagenet.json", "r") as f:
    labels_dict = json.load(f)


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        img = frame_to_PIL(frame)
        img_tensor = preprocess(img)
        img_tensor.unsqueeze_(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            class_index = int(predicted[0])
            class_name = labels_dict[str(class_index)]

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, 30)  # Coordinates for the bottom-left corner of the text
        fontScale = 1
        color = (0, 255, 0)  # White color in BGR
        thickness = 2
        lineType = 2

        # Draw the text on the frame
        frame = cv2.putText(
            frame, class_name, org, font, fontScale, color, thickness, lineType
        )
        ret, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes()

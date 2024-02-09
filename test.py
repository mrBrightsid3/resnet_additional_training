import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import json
import cv2

with open('imagenet.json', 'r') as f:
    labels_dict = json.load(f)

# Загружаем предварительно обученную модель ResNet34
model = models.resnet34(pretrained=False)

# Загружаем веса из файла .pth
state_dict = torch.load('resnet34.pth')
model.load_state_dict(state_dict)
model.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def frame_to_PIL(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap = cv2.VideoCapture(0)  # Use  0 for webcam, or replace with video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL Image
    img = frame_to_PIL(frame)

    # Preprocess the image
    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs,  1)
        class_index = int(predicted[0])
        class_name = labels_dict[str(class_index)]

    # Print the predicted class name
    print(f"Predicted class: {class_name}")

    # Display the resulting frame
    cv2.imshow('Video Stream', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

# After the loop release the cap object and destroy all windows
cap.release()
cv2.destroyAllWindows()
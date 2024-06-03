import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms.functional as F

# Define evaluation metrics calculation
def calculate_metrics(pred_boxes, pred_labels, true_boxes, true_labels, iou_threshold=0.5):
    all_true_labels = []
    all_pred_labels = []

    for i in range(len(pred_labels)):
        true_labels_set = set(true_labels[i])
        pred_labels_set = set(pred_labels[i])

        all_true_labels.extend(true_labels[i])
        all_pred_labels.extend(pred_labels[i])

    precision = precision_score(all_true_labels, all_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(all_true_labels, all_pred_labels, average='weighted', zero_division=1)
    f1 = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=1)

    return precision, recall, f1

# Define the training loop
def train_model(model, data_loader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in tqdm(data_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader)}")

# Evaluate the model
def evaluate(model, data_loader, device):
    model.eval()
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    true_boxes = []
    true_labels = []

    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            outputs = model(images)
            for i, output in enumerate(outputs):
                pred_boxes.append(output['boxes'].cpu().numpy())
                pred_labels.append(output['labels'].cpu().numpy())
                pred_scores.append(output['scores'].cpu().numpy())
                true_boxes.append(targets[i]['boxes'].cpu().numpy())
                true_labels.append(targets[i]['labels'].cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    return pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, avg_loss

# Visualize the results
def visualize(images, pred_boxes, pred_labels, true_boxes, true_labels, class_names, pred_scores, num_plots=2):
    num_plots = min(num_plots, len(images))
    for i in range(num_plots):
        image = F.to_pil_image(images[i])
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)

        # Plot predicted boxes with labels
        for box, label, score in zip(pred_boxes[i], pred_labels[i], pred_scores[i]):
            box = box.astype(np.int32)
            label_name = class_names[label - 1]
            color = 'red'
            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor=color, linewidth=2))
            ax.text(box[0], box[1], f'{label_name} {score:.2f}', color=color, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # Plot true boxes with labels
        for box, label in zip(true_boxes[i], true_labels[i]):
            box = box.astype(np.int32)
            label_name = class_names[label - 1]
            color = 'blue'
            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor=color, linewidth=2))
            ax.text(box[0], box[1], label_name, color=color, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        plt.axis('off')
        plt.show()

# Training and evaluation
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_EPOCHS = 15
BATCH_SIZE = 8
VOC_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = len(VOC_CLASSES)-1  # -1 background, 21

class VOCDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.annotation_dir = os.path.join(root, 'Annotations')
        self.image_ids = [f.split('.')[0] for f in os.listdir(self.image_dir)]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id + '.jpg')
        annotation_path = os.path.join(self.annotation_dir, image_id + '.xml')
        
        img = Image.open(img_path).convert("RGB")
        target = self.parse_voc_xml(ET.parse(annotation_path).getroot())
        
        if self.transforms:
            img = self.transforms(img)
        
        target = self.transform_target(target)
        
        return img, target

    def __len__(self):
        return len(self.image_ids)

    def parse_voc_xml(self, node):
        # Parsing the Pascal VOC XML file
        voc_dict = {}
        for elem in node:
            if elem.tag == 'object':
                obj = {}
                for e in elem:
                    if e.tag == 'name':
                        obj['name'] = e.text
                    if e.tag == 'bndbox':
                        bndbox = []
                        for bb in e:
                            bndbox.append(int(bb.text))
                        obj['bbox'] = bndbox
                voc_dict.setdefault('annotations', []).append(obj)
            else:
                voc_dict[elem.tag] = elem.text
        return voc_dict

    def transform_target(self, target):
        # Transform target to the format expected by the model
        boxes = []
        labels = []
        for obj in target['annotations']:
            bbox = obj['bbox']
            # Ensure bounding box is valid
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                boxes.append(bbox)
                labels.append(1)  # Assuming all objects are of class 1 for simplicity
        if not boxes:  # If no valid boxes, add a dummy box
            boxes.append([0, 0, 1, 1])
            labels.append(0)
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }


# Assuming dataset and dataset_test are already defined
dataset = VOCDataset(VOC_ROOT, transforms=get_transform(train=True))
dataset = Subset(dataset, range(500))  # Increase the subset size to 500 images
dataset_test = VOCDataset(VOC_ROOT, transforms=get_transform(train=False))
dataset_test = Subset(dataset_test, range(100))  # Increase the test subset size to 100 images

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

train_model(model, data_loader, optimizer, NUM_EPOCHS, DEVICE)

pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, avg_loss = evaluate(model, data_loader_test, DEVICE)
precision, recall, f1 = calculate_metrics(pred_boxes, pred_labels, true_boxes, true_labels)

print(f"Average Loss: {avg_loss}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Visualize a subset of the results
visualize([dataset_test[i][0] for i in range(len(dataset_test))], pred_boxes, pred_labels, true_boxes, true_labels, VOC_CLASSES[1:], pred_scores, num_plots=2)

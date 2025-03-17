from lib import *
from model import SSD

def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

dataset = load_json('./data/annotations/instances_val2017.json')
print("[INFO] dataset loaded: {}".format(dataset.keys()))

images = dataset["images"]
annotations = dataset["annotations"]
categories = dataset["categories"]
print("[INFO] images: {}, annotations: {}, categories: {}".format(len(images), len(annotations), len(categories)))

DATA_LABELS = ['__background__']
classes = []

for category in categories:
    while int(category['id']) - len(DATA_LABELS) > 0:
        DATA_LABELS.append("None")
    DATA_LABELS.append(category['name'])
    classes.append(category['name'])

print(DATA_LABELS.__len__())
print(classes.__len__())

cfg = {
    "num_classes": 81,#+ 1 background class
    "input_size": 300, #SSD300
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4], # Tỷ lệ khung hình cho source1->source6`
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300], # Size of default box
    "min_size": [30, 60, 111, 162, 213, 264], # Size of default box
    "max_size": [60, 111, 162, 213, 264, 315], # Size of default box
    "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]]
}

net = SSD(phase="inference", cfg=cfg)
net_weights = torch.load("./data/weights/ssd300_10.pth", map_location={"cuda:0":"cpu"})
net.load_state_dict(net_weights)

def show_predict(img_file_path):
    img = cv2.imread(img_file_path)

    color_mean = (104, 117, 123)
    input_size = 300

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    phase = "val"
    img_tensor = transform(img)

    net.eval()
    #(1, 3, 300, 300)
    input = img_tensor.unsqueeze(0)
    output = net(input)

    plt.figure(figsize=(10, 10))
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    font = cv2.FONT_HERSHEY_SIMPLEX

    detections = output.data #(1, 21, 200, 5) 5: score, cx, cy, w, h
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            cv2.rectangle(img,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          colors[i%3], 2
                          )
            display_text = "%s: %.2f"%(classes[i-1], score)
            cv2.putText(img, display_text, (int(pt[0]), int(pt[1])),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            j += 1
    
    # cv2.imshow("Result", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    img_file_path = "./data/val2017/000000000802.jpg"
    show_predict(img_file_path)
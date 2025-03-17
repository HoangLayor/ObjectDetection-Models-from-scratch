from lib import *
from extract_annotation import ExtractAnno
from model import SSD
from multibox_loss import MultiBoxLoss
from dataset import MyDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 8

LABELS = 1.0
BBOX = 1.0
root = "./data/val2017"
color_mean = (104, 117, 123)
input_size = (300, 300)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] device: ", device)
torch.backends.cudnn.benchmark = True

def my_collate_fn(batch):
    _images = []
    _annotations = []

    for sample in batch:
        _images.append(sample[0])
        _annotations.append(torch.FloatTensor(np.array(sample[1])))

    _images = torch.stack(_images, dim=0)
    return _images, _annotations

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

annoExtractor = ExtractAnno("./data/annotations/instances_val2017.json")
classes = annoExtractor.classes
img_list, anno_list = annoExtractor.extract()

split = train_test_split(img_list, anno_list, test_size=0.2, random_state=42)
trainImages, testImages, trainAnno, testAnno = split

print("[INFO] training samples: ", len(trainImages))
print("[INFO]  testing samples: ", len(testImages))

transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize(input_size),
	transforms.ToTensor(),
	transforms.Normalize(mean=MEAN, std=STD)
])

train_dataset = MyDataset(root, trainImages, trainAnno, transform=transform)
test_dataset = MyDataset(root, testImages, testAnno, transform=transform)
print("[INFO] total training samples: {}...".format(len(train_dataset)))
print("[INFO] total  testing samples: {}...".format(len(test_dataset)))

trainLoader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, collate_fn=my_collate_fn)
testLoader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, collate_fn=my_collate_fn)

print(train_dataset[0])
print("!"*10)
print(train_dataset[0])
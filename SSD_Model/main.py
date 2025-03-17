from lib import *
from extract_annotation import ExtractAnno
from model import SSD
from multibox_loss import MultiBoxLoss
from dataset import MyDataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
root = "./data/val2017"
color_mean = (104, 117, 123)
input_size = (300, 300)

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] device:", device)
torch.backends.cudnn.benchmark = True

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
print("[INFO] total test samples: {}...".format(len(test_dataset)))

trainLoader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate_fn)
testLoader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate_fn)

dataloader_dict = {"train": trainLoader, "val": testLoader}
cfg = {
    "num_classes": 81, #+ 1 background class
    "input_size": 300, #SSD300
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4], # Tỷ lệ khung hình cho source1->source6`
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300], # Size of default box
    "min_size": [30, 60, 111, 162, 213, 264], # Size of default box
    "max_size": [60, 111, 162, 213, 264, 315], # Size of default box
    "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]]
}

net = SSD(phase="inference", cfg=cfg)
net_weights = torch.load("./data/weights/ssd300_30.pth", map_location={"cuda:0": "cpu"})
net.load_state_dict(net_weights)

# net = SSD(phase="train", cfg=cfg)
# vgg_weights = torch.load("./data/weights/vgg16_reducedfc.pth")
# net.vgg.load_state_dict(vgg_weights)

# # He init
# net.extras.apply(weights_init)
# net.loc.apply(weights_init)
# net.conf.apply(weights_init)

# MultiBoxLoss
criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=DEVICE)
# optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)

# training, validation
def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    # move network to GPU
    net.to(device)

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    for epoch in range(num_epochs+1):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        print("---"*20)
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("---"*20)
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print("(Training)")
            else:
                if (epoch+1) % 10 == 0:
                    net.eval() 
                    print("---"*10)
                    print("(Validation)")
                else:
                    continue
            
            for images, targets in dataloader_dict[phase]:
                # move to GPU
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]
                
                # init optimizer
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase=="train"):
                    outputs = net(images)
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        loss.backward() # calculate gradient
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                        optimizer.step() # update parameters

                        if (iteration % 10) == 0:
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print("Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec".format(iteration, loss.item(), duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
            
        t_epoch_end = time.time()
        print("---"*20)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))           
        print("Duration: {:.4f} sec".format(t_epoch_end - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {"epoch": epoch+1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("./data/ssd_logs.csv")
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), "./data/weights/ssd300_" + str(epoch+1) + ".pth")

num_epochs = 100
train_model(net, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)
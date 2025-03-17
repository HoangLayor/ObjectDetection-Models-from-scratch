from lib import *

def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

class ExtractAnno():
    def __init__(self, anno_file: str) -> None:
        '''
        Extract annotation information from json file
        Args:
            anno_file: str, path to json file
        '''
        self.dataset = load_json(anno_file)
        print("[INFO] dataset loaded: {}".format(anno_file))

        self.images, self.annotations, self.categories = self.dataset["images"], self.dataset["annotations"], self.dataset["categories"]
        print("[INFO] images: {}, annotations: {}, categories: {}".format(len(self.images), len(self.annotations), len(self.categories)))

        # Extract class labels
        self.DATA_LABELS = ['__background__']
        self.classes = []
        for category in self.dataset["categories"]:
            while int(category['id']) - len(self.DATA_LABELS) > 0:
                self.DATA_LABELS.append("None")

            self.DATA_LABELS.append(category['name'])
            self.classes.append(category['name'])

        print(f"[INFO] classes: {self.classes}")

    def extract(self):
        print("[INFO] loading dataset...")

        id_list = []
        img_list = []
        anno_json = {}
        anno_list = []
        background_list = []

        # Extract image_list
        for img in self.dataset["images"]:
            image_id = img["id"]
            file_name = img["file_name"]
            height = img["height"]
            width = img["width"]
            id_list.append(image_id)
            img_list.append((image_id, file_name, height, width))
            anno_json[image_id] = []

        # Extract annotation_list
        for anno in self.dataset["annotations"]:
            image_id = anno["image_id"]
            label = self.DATA_LABELS[anno["category_id"]] # label_id -> label
            label_id = self.classes.index(label) # label -> index of label
            x_min, y_min, w, h = anno["bbox"]
            anno_json[image_id].append([x_min, y_min, x_min+w, y_min+h, label_id]) # [x_min, y_min, x_max, y_max, label_id]

        # Annotation list
        for key, value in anno_json.items():
            anno_list.append((key, np.array(value)))
            if value == []:
                background_list.append(key)

        id_list = sorted(id_list)
        anno_list = sorted(anno_list, key=lambda x: x[0])
        img_list = sorted(img_list, key=lambda x: x[0])

        img_list = [img for img in img_list if img[0] not in background_list]
        anno_list = [anno for anno in anno_list if anno[0] not in background_list]

        print("[INFO] loading dataset complete")
        print("[INFO] number of images: ", len(img_list))
        print("[INFO] number of annotations: ", len(anno_list))
        print("[INFO] number of background images: ", len(background_list))
        
        return img_list, anno_list
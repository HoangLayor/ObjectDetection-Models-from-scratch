from lib import *

class DefBox():
    def __init__(self, cfg):
        self.img_size = cfg["input_size"]
        self.feature_maps = cfg["feature_maps"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.steps = cfg["steps"]

    def create_defbox(self):
        defbox_list = []

        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.img_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                s_k = self.min_size[k] / self.img_size
                defbox_list += [cx, cy, s_k, s_k]

                s_k_ = np.sqrt(s_k * (self.max_size[k] / self.img_size))
                defbox_list += [cx, cy, s_k_, s_k_]

                for ar in self.aspect_ratios[k]:
                    defbox_list += [cx, cy, s_k * np.sqrt(ar), s_k / np.sqrt(ar)]
                    defbox_list += [cx, cy, s_k / np.sqrt(ar), s_k * np.sqrt(ar)]

        output = torch.Tensor(defbox_list).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output
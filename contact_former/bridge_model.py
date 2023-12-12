import torch
import torch.nn as nn
import numpy as np
from collections import Counter

from util.translate_obj_bbox import *


class BridgeModel(nn.Module):
    def __init__(self, atiss_model, cf_model, datatype, num_classes, device):
        super(BridgeModel, self).__init__()
        self.atiss_model = atiss_model
        self.cf_model = cf_model
        self.device = device
        self.datatype = datatype
        self.num_classes = num_classes
        
        # Freeze ContactFormer model
        for param in self.cf_model.parameters():
            param.requires_grad = False
    
    def forward(self, given_objs, given_cats, mask):
        device = self.device
        bs = given_objs.shape[0]
        human_pose = given_objs[:, 0]
        chosen_indices = torch.randint(low=0, high=1024, size=(655,))
        human_pose = human_pose[:, chosen_indices]

        z = torch.tensor(np.random.normal(0, 1, (bs, 256)).astype(np.float32)).to(self.device)
        contact_points = self.cf_model.posa.decoder(z, human_pose)
        contact_points = contact_points.argmax(dim=-1)

        # Get bounding box for human
        default_translation, default_size = translate_target_obj_to_bbox(human_pose)
        
        # Get the most frequent category
        translated_cp = []
        for cp_batch in contact_points:
            cp = []
            for point in cp_batch:
                cp.append(self._lookup_table(point))
            
            translated_cp.append(cp)
        translated_cp = torch.tensor(translated_cp)
        
        obj_cat = []
        for idx, cp_batch in enumerate(translated_cp):
            counter = Counter(cp_batch.tolist())
            if len(counter) == 1:
                cat = 0
                translation = default_translation
                size = default_size
            else:
                cat = counter.most_common()[1][0]
                indices = [cp_batch==cat]
                points = human_pose[idx][indices]
                translation = points.mean(dim=0)
                size = default_size
            
        # Calculate number of objects
        num_obj = len(mask[0])
        for idx in range(1, len(mask[0])):
            if mask[0][idx] == 0:
                num_obj = idx
                break

        # Compute boxes for ATISS model
        bs, _, _, _ = given_objs.shape
        translations, sizes = translate_objs_to_bbox(given_objs[:, :num_obj], mask[:, :num_obj])
        boxes = {}
        boxes['class_labels'] = given_cats[:, :num_obj]
        boxes['translations'] = translations.to(device)
        boxes['translations'][:, 0] = translation.to(device)
        boxes['sizes'] = sizes.to(device)
        boxes['sizes'][:, 0] = size.to(device)
        boxes['angles'] = torch.zeros((bs, num_obj, 1)).to(device)

        # Fill in input boxes attribute
        boxes['room_layout'] = torch.ones((bs, 1, 64, 64)).to(device)
        boxes['lengths'] = torch.ones(1).to(device)
        boxes['class_labels_tr'] = torch.ones((bs, 1, self.num_classes)).to(device)
        boxes['translations_tr'] = torch.ones((bs, 1, 3)).to(device)
        boxes['sizes_tr'] = torch.ones((bs, 1, 3)).to(device)
        boxes['angles_tr'] = torch.ones((bs, 1, 1)).to(device)

        output_obj = self.atiss_model(boxes)
        return output_obj

    @property
    def _pred_subset_to_mpcat40(self):
        return {
            0: "void",  # void
            1: "wall",  # wall
            2: "floor",  # floor
            3: "chair",  # chair
            4: "sofa", # sofa
            5: "table",  # table
            6: "bed", # bed
            7: "stool", # stool
        }
    
    @property
    def _protext_cat(self):
        return {
            "chair": 1,
            "table": 2,
            "cabinet": 3,
            "sofa": 4,
            "bed": 5,
            "chest_of_drawers": 6,
            "chest": 6,
            "stool": 7,
            "tv_monitor": 8,
            "tv": 8,
            "lighting": 9,
            "shelving": 10,
            "seating": 11,
            "furniture": 12,
            "human": 0,
        }
    
    @property
    def _humanise_cat(self):
        return {
            "bed": 1,		# bed
            "sofa": 2,  		# sofa
            "table": 3,		# table
            "door": 4,  		# door
            "desk": 5,		# desk
            "refrigerator": 6, 		# refrigerator
            "chair": 7,
            "counter": 8,
            "bookshelf": 9,
            "cabinet": 10,
            "human": 0
        }


    def _lookup_table(self, index):
        if self.datatype == "proxd":
            cat = self._pred_subset_to_mpcat40[index.item()]
            if cat not in self._protext_cat:
                return -1
            return self._protext_cat[cat]
        
        if self.datatype == "humanise":
            cat = self._pred_subset_to_mpcat40[index.item()]
            if cat not in self._humanise_cat:
                return -1
            return self._humanise_cat[cat]
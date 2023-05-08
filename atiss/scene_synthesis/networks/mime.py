import torch
import torch.nn as nn
from .autoregressive_transformer import BaseAutoregressiveTransformer

class MIME(BaseAutoregressiveTransformer):
    def __init__(self, input_dims, hidden2output, feature_extractor, config):
        super().__init__(input_dims, hidden2output, feature_extractor, config)
        # Embedding to be used for the empty/mask token
        self.register_parameter(
            "empty_token_embedding", nn.Parameter(torch.randn(1, 528))
        )
        self.fc_room_f = nn.Linear(
            self.feature_extractor.feature_size, 528
        )
        n_contacts = 1
        hidden_dims = config.get("hidden_dims", 768)
        self.fc = nn.Linear(528, hidden_dims)
        self.contact_fc = nn.Linear(n_contacts, 16, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=528, nhead=config.get("n_heads", 12),
            dim_feedforward=config.get("feed_forward_dimensions", 3072), activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.get("n_layers", 6))
    
    def forward(self, sample_params):
        # Unpack the sample_params
        contact_labels = sample_params["contact_labels"]
        class_labels = sample_params["class_labels"]
        translations = sample_params["translations"]
        sizes = sample_params["sizes"]
        angles = sample_params["angles"]
        room_layout = sample_params["room_layout"]
        B, _, _ = class_labels.shape

        # Apply the contact features
        contact_f = self.contact_fc(contact_labels)
        # Apply the positional embeddings only on bboxes that are not the start
        # token
        class_f = self.fc_class(class_labels)
        # Apply the positional embedding along each dimension of the position
        # property
        pos_f_x = self.pe_pos_x(translations[:, :, 0:1])
        pos_f_y = self.pe_pos_y(translations[:, :, 1:2])
        pos_f_z = self.pe_pos_z(translations[:, :, 2:3])
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

        size_f_x = self.pe_size_x(sizes[:, :, 0:1])
        size_f_y = self.pe_size_y(sizes[:, :, 1:2])
        size_f_z = self.pe_size_z(sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        angle_f = self.pe_angle_z(angles)
        X = torch.cat([contact_f, class_f, pos_f, size_f, angle_f], dim=-1)

        start_symbol_f = self.start_symbol_features(B, room_layout)
        # Concatenate with the mask embedding for the start token
        X = torch.cat([
            start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
        ], dim=1)
        X = self.fc(X)

        # Compute the features using causal masking
        F = self.transformer_encoder(X)
        return self.hidden2output(F[:, 1:2], sample_params)

    def _encode(self, boxes, room_mask):
        class_labels = boxes["class_labels"]
        translations = boxes["translations"]
        sizes = boxes["sizes"]
        angles = boxes["angles"]
        B, _, _ = class_labels.shape

        if class_labels.shape[1] == 1:
            start_symbol_f = self.start_symbol_features(B, room_mask)
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1)
            ], dim=1)
        else:
            # Apply the positional embeddings only on bboxes that are not the
            # start token
            class_f = self.fc_class(class_labels[:, 1:])
            # Apply the positional embedding along each dimension of the
            # position property
            pos_f_x = self.pe_pos_x(translations[:, 1:, 0:1])
            pos_f_y = self.pe_pos_y(translations[:, 1:, 1:2])
            pos_f_z = self.pe_pos_z(translations[:, 1:, 2:3])
            pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

            size_f_x = self.pe_size_x(sizes[:, 1:, 0:1])
            size_f_y = self.pe_size_y(sizes[:, 1:, 1:2])
            size_f_z = self.pe_size_z(sizes[:, 1:, 2:3])
            size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

            angle_f = self.pe_angle_z(angles[:, 1:])
            X = torch.cat([class_f, pos_f, size_f, angle_f], dim=-1)

            start_symbol_f = self.start_symbol_features(B, room_mask)
            # Concatenate with the mask embedding for the start token
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
            ], dim=1)
        X = self.fc(X)
        F = self.transformer_encoder(X, length_mask=None)[:, 1:2]

        return F

    def autoregressive_decode(self, boxes, room_mask):
        class_labels = boxes["class_labels"]

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)
        # Sample the class label for the next bbbox
        class_labels = self.hidden2output.sample_class_labels(F)
        # Sample the translations
        translations = self.hidden2output.sample_translations(F, class_labels)
        # Sample the angles
        angles = self.hidden2output.sample_angles(
            F, class_labels, translations
        )
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_labels, translations, angles
        )

        return {
            "class_labels": class_labels,
            "translations": translations,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def generate_boxes(self, room_mask, max_boxes=32, device="cpu"):
        boxes = self.start_symbol(device)
        for i in range(max_boxes):
            box = self.autoregressive_decode(boxes, room_mask=room_mask)

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:
                break

        return {
            "class_labels": boxes["class_labels"].to("cpu"),
            "translations": boxes["translations"].to("cpu"),
            "sizes": boxes["sizes"].to("cpu"),
            "angles": boxes["angles"].to("cpu")
        }

    def autoregressive_decode_with_class_label(
        self, boxes, room_mask, class_label
    ):
        class_labels = boxes["class_labels"]
        B, _, C = class_labels.shape

        # Make sure that everything has the correct size
        assert len(class_label.shape) == 3
        assert class_label.shape[0] == B
        assert class_label.shape[-1] == C

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Sample the translations conditioned on the query_class_label
        translations = self.hidden2output.sample_translations(F, class_label)
        # Sample the angles
        angles = self.hidden2output.sample_angles(
            F, class_label, translations
        )
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_label, translations, angles
        )

        return {
            "class_labels": class_label,
            "translations": translations,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def add_object(self, room_mask, class_label, boxes=None, device="cpu"):
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode_with_class_label(
            boxes=boxes,
            room_mask=room_mask,
            class_label=class_label
        )

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    @torch.no_grad()
    def complete_scene(
        self,
        boxes,
        room_mask,
        max_boxes=100,
        device="cpu"
    ):
        boxes = dict(boxes.items())

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        for i in range(max_boxes):
            box = self.autoregressive_decode(boxes, room_mask=room_mask)

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:
                break

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    def autoregressive_decode_with_class_label_and_translation(
        self,
        boxes,
        room_mask,
        class_label,
        translation
    ):
        class_labels = boxes["class_labels"]
        B, _, C = class_labels.shape

        # Make sure that everything has the correct size
        assert len(class_label.shape) == 3
        assert class_label.shape[0] == B
        assert class_label.shape[-1] == C

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Sample the angles
        angles = self.hidden2output.sample_angles(F, class_label, translation)
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_label, translation, angles
        )

        return {
            "class_labels": class_label,
            "translations": translation,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def add_object_with_class_and_translation(
        self,
        boxes,
        room_mask,
        class_label,
        translation,
        device="cpu"
    ):
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)


        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode_with_class_label_and_translation(
            boxes=boxes,
            class_label=class_label,
            translation=translation,
            room_mask=room_mask
        )

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    @torch.no_grad()
    def distribution_classes(self, boxes, room_mask, device="cpu"):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())
        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)
        return self.hidden2output.pred_class_probs(F)

    @torch.no_grad()
    def distribution_translations(
        self,
        boxes,
        room_mask, 
        class_label,
        device="cpu"
    ):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Concatenate to the given input (that's why we shallow copy in the
        # beginning of this method
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Get the dmll params for the translations
        return self.hidden2output.pred_dmll_params_translation(
            F, class_label
        )

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .medsam.segment_anything import sam_model_registry
from .cemb import CEmb



class CEmbSam(nn.Module):

    def __init__(
            self, 
            model_type, 
            checkpoint,
            num_feature,
            emb_classes
    ) -> None:
        """
        SAM with Condition Embedding block.

        Arguments:
            model_type (str): 
            checkpoint (): 
            num_feature (int): 
            emb_classes (int): 
        """
        super().__init__()
        self.sam_model = sam_model_registry[model_type](
            checkpoint=checkpoint
        )
        self.con_block = CEmb(
            num_features=num_feature,
            emb_classes=emb_classes
        )

    def forward(self, image, bbox, wcode):

        image_embeddings = self.sam_model.image_encoder(image)

        image_embeddings = self.con_block(image_embeddings, wcode)

        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=None,
            boxes=bbox,
            masks=None
        )

        mask_predictions, _ = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings,  # (B, 256, 64, 64)
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        return mask_predictions

if __name__ == "__main__":
    pass
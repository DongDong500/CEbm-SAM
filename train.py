import argparse
import os
import json
from datetime import datetime
import traceback
import pandas as pd
import numpy as np
import monai
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from typing import Any, Iterable
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from torch.utils.tensorboard import SummaryWriter

from method.medsam.segment_anything import sam_model_registry
from method.medsam.segment_anything.utils.transforms import ResizeLongestSide
from method.medsam.utils.dataset import MedSamDataset, UltraSamDataset
from method import CEmbSam

from utils.utils import earlystop, seg_metrics, email, utils
from tabulate import tabulate

from method.unet import Unet



def dice_score(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()

    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    
    return dice

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv", type=str, required=True, help="Path to the CSV file"
    )
    parser.add_argument(
        "--image_col",
        type=str,
        default="image",
        help="Name of the column on the dataframe that holds the image file names",
    )
    parser.add_argument(
        "--mask_col",
        type=str,
        default="mask",
        help="the name of the column on the dataframe that holds the mask file names",
    )
    parser.add_argument(
        "--image", 
        type=str, 
        required=False,
        default="/home/dongik/datasets", 
        help="Path to the input image directory"
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=False,
        default="/home/dongik/datasets",
        help="Path to the ground truth mask directory",
    )
    parser.add_argument(
        "--num_epochs", type=int, required=False, default=100, help="number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, required=False, default=3e-4, help="learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=8, help="batch size"
    )
    parser.add_argument(
        "-k",
        type=int,
        default=None,
        required=False,
        help="Number of folds for cross validation",
    )  
    parser.add_argument(
        "--model_type", type=str, required=False, default="vit_b"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False, help="Path to SAM checkpoint"
    )
    parser.add_argument(
        "--save_pth", type=str, required=True, help="Path to save checkpoint"
    )
    parser.add_argument(
        "--train_model_type", type=str, choices=["CEmbSam"], default="CEmb-Sam",
        help="choose model type to train"
    )
    parser.add_argument(
        "--emb_classes", type=int, required=True, default=3,
        help=""
    )

    return parser.parse_args()

class TrainCEmbSam:
    BEST_VAL_LOSS = float("inf")
    BEST_EPOCH = 0

    def __init__(
            self,
            lr: float = 3e-4,
            batch_size: int = 4,
            epochs: int = 100,
            device: str = "cuda",
            model_type: str = "vit_b",
            image_dir="data/image_dir",
            mask_dir="data/image_dir",
            checkpoint: str = "work_dir/SAM/sam_vit_b_01ec64.pth",
            save_pth: str = "work_dir/uSAM-result",
            emb_classes: int = 7
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sam_checkpoint_dir = checkpoint
        self.model_type = model_type
        self.save_pth = save_pth
        self.emb_classes = emb_classes
    
    def __call__(
            self,
            train_df, 
            test_df, 
            val_df, 
            image_col, 
            mask_col
    ):
        """Entry method
        prepare `dataset` and `dataloader` objects

        """
        print("[INFO] Train CEmb-Sam")

        train_ds = UltraSamDataset(
            train_df,
            image_col,
            mask_col,
            self.image_dir,
            self.mask_dir,
        )
        val_ds = UltraSamDataset(
            val_df,
            image_col,
            mask_col,
            self.image_dir,
            self.mask_dir,
        )
        test_ds = UltraSamDataset(
            test_df,
            image_col,
            mask_col,
            self.image_dir,
            self.mask_dir,
        )
        # Define dataloaders
        train_loader = DataLoader(
            dataset=train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            dataset=val_ds, batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            dataset=test_ds, batch_size=self.batch_size, shuffle=False
        )
        # Get the model
        model = self.get_model()

        # Train and evaluate model
        self.train(model, train_loader, val_loader)
        # Evaluate model on test data
        loss, dice_score, metric = self.test(model, test_loader, desc="Testing")

        del model
        torch.cuda.empty_cache()

        self.BEST_EPOCH = 0
        self.BEST_VAL_LOSS = float("inf")

        return dice_score, metric
        
    def get_model(self):
        sam_model = CEmbSam(
            model_type=self.model_type,
            checkpoint=self.sam_checkpoint_dir,
            num_feature=256,
            emb_classes=self.emb_classes
        )
        if torch.cuda.device_count() > 1:
            print("cuda multiple GPUs")
            sam_model = nn.DataParallel(sam_model)
        else:
            print("cuda single GPUs")
        sam_model.to(self.device)

        return sam_model

    @torch.inference_mode()
    def evaluate(self, model, val_loader, desc="Validating") -> float:
        """Perform evaluation on test or validation data
        Args:
            model (_type_): pytorch model
            val_loader (_type_): dataloader
            desc (str, optional): _description_. Defaults to "Validating".

        Returns:
            np.array: (mean validation loss, mean validation dice)
        """
        seg_loss = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )

        progress_bar = tqdm(
            val_loader, 
            total=len(val_loader),
            ascii=" =",
            leave=False
        )
        val_loss = []
        val_dice = []

        for image, mask, bbox, wcode in progress_bar:
            image = image.to(self.device)
            mask = mask.to(self.device)
            # resize image to 1024 by 1024
            image = TF.resize(image, (1024, 1024), antialias=True)
            H, W = mask.shape[-2], mask.shape[-1]

            sam_trans = ResizeLongestSide(model.sam_model.image_encoder.img_size)

            box = sam_trans.apply_boxes(bbox, (H, W))
            box_tensor = torch.as_tensor(box, dtype=torch.float, device=self.device)

            # Get predictioin mask

            image_embeddings = model.sam_model.image_encoder(image)  # (B,256,64,64)

            ###
            ### condition embedding block
            ###
            image_embeddings = model.cemb(
                image_embeddings.to(self.device), 
                wcode.to(self.device)
            )

            sparse_embeddings, dense_embeddings = model.sam_model.prompt_encoder(
                points=None,
                boxes=box_tensor,
                masks=None,
            )

            mask_predictions, _ = model.sam_model.mask_decoder(
                image_embeddings=image_embeddings.to(self.device),  # (B, 256, 64, 64)
                image_pe=model.sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            mask_predictions = (mask_predictions > 0.5).float()

            # get the dice loss
            loss = seg_loss(mask_predictions, mask)
            dice = dice_score(mask_predictions, mask)

            val_loss.append(loss.detach().item())
            val_dice.append(dice.detach().item())

            # Update the progress bar
            progress_bar.set_description(f"[{desc}]")
            progress_bar.set_postfix(
                eval_loss=np.mean(val_loss), eval_dice=np.mean(val_dice)
            )
            progress_bar.update()
        return np.mean(val_loss), np.mean(val_dice)
    
    @torch.inference_mode()
    def test(self, model, val_loader, desc="Testing"):
        """Perform evaluation on test or validation data
        Args:
            model (_type_): pytorch model
            val_loader (_type_): dataloader
            desc (str, optional): _description_.

        Returns:
            float: mean validation loss
        """
        metrics = seg_metrics.StreamSegMetrics(
            n_classes=2
        )

        seg_loss = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )
        progress_bar = tqdm(
            val_loader, 
            total=len(val_loader),
            ascii=" =",
            leave=False
        )
        val_loss = []
        dice_scores = []

        for image, mask, bbox, wcode in progress_bar:
            image = image.to(self.device)
            mask = mask.to(self.device)
            # resize image to 1024 by 1024
            image = TF.resize(image, (1024, 1024), antialias=True)
            H, W = mask.shape[-2], mask.shape[-1]
            sam_trans = ResizeLongestSide(model.sam_model.image_encoder.img_size)

            box = sam_trans.apply_boxes(bbox, (H, W))
            box_tensor = torch.as_tensor(box, dtype=torch.float, device=self.device)
            # Get predictioin mask

            image_embeddings = model.sam_model.image_encoder(image)  # (B,256,64,64)
            ###
            ### condition embedding block
            ###
            image_embeddings = model.cemb(
                image_embeddings.to(self.device), 
                wcode.to(self.device)
            )

            sparse_embeddings, dense_embeddings = model.sam_model.prompt_encoder(
                points=None,
                boxes=box_tensor,
                masks=None,
            )

            mask_predictions, _ = model.sam_model.mask_decoder(
                image_embeddings=image_embeddings.to(self.device),  # (B, 256, 64, 64)
                image_pe=model.sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )

            # get the dice loss
            loss = seg_loss(mask_predictions, mask)

            mask_predictions = (mask_predictions > 0.5).float()
            dice = dice_score(mask_predictions, mask)

            val_loss.append(loss.item())
            dice_scores.append(dice.detach().item())

            mask = (mask > 0).float()
            metrics.update(
                label_preds=mask_predictions.detach().cpu().numpy(),
                label_trues=mask.detach().cpu().numpy()
            )

            # Update the progress bar
            progress_bar.set_description(f"[{desc}]")
            progress_bar.set_postfix(
                eval_loss=np.mean(val_loss), eval_dice=np.mean(dice_scores)
            )
            progress_bar.update()
        return np.mean(val_loss), np.mean(dice_scores), metrics.get_results()

    def train(self, model, train_loader: Iterable, val_loader: Iterable, logg=True):
        """Train the model"""

        sam_trans = ResizeLongestSide(model.sam_model.image_encoder.img_size)
        writer = SummaryWriter(
            log_dir=os.path.join(
                self.save_pth,
                "runs"
            )
        )

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.01, verbose=True
        )
        seg_loss = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )
        model.train()
        for epoch in range(self.epochs):
            epoch_loss = []
            epoch_dice = []
            progress_bar = tqdm(
                train_loader, 
                total=len(train_loader),
                ascii=" =",
                leave=False
            )
            for image, mask, bbox, wcode in progress_bar:
                image = image.to(self.device)
                mask = mask.to(self.device)
                wcode = wcode.to(self.device)
                # resize image to 1024 by 1024
                image = TF.resize(image, (1024, 1024), antialias=True)
                H, W = mask.shape[-2], mask.shape[-1]

                box = sam_trans.apply_boxes(bbox, (H, W))
                box_tensor = torch.as_tensor(box, dtype=torch.float, device=self.device)

                # Get predictioin mask
                with torch.inference_mode():
                    image_embeddings = model.sam_model.image_encoder(image)  # (B,256,64,64)

                    sparse_embeddings, dense_embeddings = model.sam_model.prompt_encoder(
                        points=None,
                        boxes=box_tensor,
                        masks=None,
                    )
                ###
                ### condition embedding block
                ###
                image_embeddings = model.cemb(
                    x=image_embeddings.clone().to(self.device), 
                    y=wcode.to(self.device)
                )
                
                mask_predictions, _ = model.sam_model.mask_decoder(
                    image_embeddings=image_embeddings.to(
                        self.device
                    ),  # (B, 256, 64, 64)
                    image_pe=model.sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                )
                
                # Calculate loss
                loss = seg_loss(mask_predictions, mask)
                
                mask_predictions = (mask_predictions > 0.5).float() # (B, 1, 256, 256) float32
                dice = dice_score(mask_predictions, mask)

                epoch_loss.append(loss.detach().item())
                epoch_dice.append(dice.detach().item())

                # empty gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.set_description(f"[Train] {epoch+1}/{self.epochs}")
                progress_bar.set_postfix(
                    loss=np.mean(epoch_loss), dice=np.mean(epoch_dice)
                )
                progress_bar.update()
            # Evaluate every two epochs
            if epoch % 2 == 0:
                validation_loss, validation_dice = self.evaluate(
                    model, val_loader, desc=f"Validating"
                )
                scheduler.step(torch.tensor(validation_loss))

                if self.early_stopping(model, validation_loss, epoch):
                    print(f"[INFO:] Early Stopping!!")
                    break

            if logg:
                writer.add_scalars(
                    "loss",
                    {
                        "train": round(np.mean(epoch_loss), 4),
                        "val": round(validation_loss, 4),
                    },
                    epoch,
                )

                writer.add_scalars(
                    "dice",
                    {
                        "train": round(np.mean(epoch_dice), 4),
                        "val": round(validation_dice, 4),
                    },
                    epoch,
                )

    def save_model(self, model, model_name=None):
        date_postfix = datetime.now().strftime("%Y-%m-%d-%H-%S")
        if model_name is None:
            model_name = f"ultrasam_finetune_{date_postfix}.pth"
        save_path = os.path.join(
            self.save_pth,
            "finetune_weights"
        )

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f"[INFO:] Saving model to {os.path.join(save_path,model_name)}")
        
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), os.path.join(save_path, model_name))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, model_name))

    def early_stopping(
        self,
        model,
        val_loss: float,
        epoch: int,
        patience: int = 10,
        min_delta: int = 0.001,
    ):
        """Helper function for model training early stopping
        Args:
            val_loss (float): _description_
            epoch (int): _description_
            patience (int, optional): _description_. Defaults to 10.
            min_delta (int, optional): _description_. Defaults to 0.01.
        """
        
        if self.BEST_VAL_LOSS - val_loss >= min_delta:
            print(
                f"[INFO:] Validation loss improved from {self.BEST_VAL_LOSS:.4f} to {val_loss:.4f}"
            )
            self.BEST_VAL_LOSS = val_loss
            self.BEST_EPOCH = epoch
            self.save_model(model)
            return False

        if (
            self.BEST_VAL_LOSS - val_loss < min_delta
            and epoch - self.BEST_EPOCH >= patience
        ):
            self.save_model(model, "ultrasam_finetune_final.pth")
            return True
        return False

class CrossValidate(TrainCEmbSam):
    
    def __init__(
        self,
        lr: float = 3e-4,
        batch_size: int = 4,
        epochs: int = 100,
        device: str = "cuda:0",
        model_type: str = "vit_b",
        image_dir="data/image_dir",
        mask_dir="data/image_dir",
        checkpoint: str = "work_dir/SAM/sam_vit_b_01ec64.pth",
        save_pth: str = "work_dir/uSAM-result"
    ):
        super().__init__(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            model_type=model_type,
            image_dir=image_dir,
            mask_dir=mask_dir,
            checkpoint=checkpoint,
            save_pth=save_pth
        )

    def __call__(self, train_df, test_df, image_col, mask_col, k: int = 5) -> Any:
        """Performs kfold cross validation
        Args:
            k (int, optional): Fold size. Defaults to 5.
        """
        # Define the cross-validation splitter
        kf = KFold(n_splits=k, shuffle=True)
        # loop over each fold
        fold_scores = {}
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            print(f"Cross validating for fold {fold + 1}")
            # Define training and validation sets for this fold
            f_train_df = train_df.iloc[train_idx]
            f_val_df = train_df.iloc[val_idx]

            dice_score, metric = super().__call__(
                f_train_df, test_df, f_val_df, image_col, mask_col
            )

            fold_scores[f"fold_{fold + 1}_mean_dice"] = dice_score
            fold_scores[f"fold_{fold + 1}_metrics"] = {
                "Precision" : {
                    "background" : metric['Class Precision'][0],
                    "RoI" : metric['Class Precision'][1]
                },
                "Recall" : {
                    "background" : metric['Class Recall'][0],
                    "RoI" : metric['Class Recall'][1]
                },
                "Specificity" : {
                    "background" : metric['Class Specificity'][0],
                    "RoI" : metric['Class Specificity'][1]
                },
                "IoU" : {
                    "background" : metric['Class IoU'][0],
                    "RoI" : metric['Class IoU'][1]
                },
                "Dice score" : {
                    "background" : metric['Class Dice'][0],
                    "RoI" : metric['Class Dice'][1]
                }
            }

        return fold_scores
    
def main(args):
    
    try:
        train_df = pd.read_csv(
            os.path.join(args.csv, "train.csv")
        )
        val_df = pd.read_csv(
            os.path.join(args.csv, "val.csv")
        )
        test_df = pd.read_csv(
            os.path.join(args.csv, "test.csv")
        )
    except FileNotFoundError:
        print(f"{args.csv} does not exist")

    # split the dataset into train and test set
    # train_df, test_df = train_test_split(df, train_size=0.8, random_state=2023)
    # if `k` argument is specified, run the cross validation
    if args.k:
        print(f"[INFO] Starting {args.k} fold cross validation ....")
        if args.k < 5:
            raise ValueError("K should be a value greater than or equal to 5")

        raise NotImplementedError
    
        cross_validate = CrossValidate(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.num_epochs,
            image_dir=args.image,
            mask_dir=args.mask,
            checkpoint=args.checkpoint,
            save_pth=args.save_pth
        )
        scores = cross_validate(
            train_df, test_df, args.image_col, args.mask_col, args.k
        )
        # write cross-validation scores to file
        with open(os.path.join(args.save_pth, "medsam.json"), "w") as f:
            json.dump(scores, f)
    # if `k` is not specified, normal training mode
    if not args.k:
        print(f"[INFO] Starting training for {args.num_epochs} epochs ....")
        
        # save result
        rst = {

        }
        
        if args.train_model_type == "Unet":
            train = TrainUnet(
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.num_epochs,
                image_dir=args.image,
                mask_dir=args.mask,
                checkpoint=args.checkpoint,
                save_pth=args.save_pth,
            )
        elif args.train_model_type == "MedSam":
            train = TrainMedSam(
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.num_epochs,
                image_dir=args.image,
                mask_dir=args.mask,
                checkpoint=args.checkpoint,
                save_pth=args.save_pth,
            )
        elif args.train_model_type == "CEmbSam":
            train = TrainCEmbSam(
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.num_epochs,
                image_dir=args.image,
                mask_dir=args.mask,
                checkpoint=args.checkpoint,
                save_pth=args.save_pth,
            )
        else:
            raise NotImplementedError
        
        dice_score, metric = train(
            train_df, test_df, val_df, args.image_col, args.mask_col
        )

        rst[args.train_model_type] = {
            "dice_score" : dice_score,
            "metric" : metric
        }
        print(f"[INFO] dice: {dice_score} metric: {metric}")

        return rst



if __name__ == "__main__":

    total_time = datetime.now()
    
    # set up parser
    args = get_argparser()
    if not os.path.exists(args.save_pth):
        os.makedirs(args.save_pth)

    try:
        is_error = False
        metric = main(args)

    except KeyboardInterrupt:
        is_error = True
        print("KeyboardInterrupt: Stop !!!")

    except Exception as e:
        is_error = True
        print("Error:", e)
        print(traceback.format_exc())
    
    total_time = datetime.now() - total_time
    print(f"[INFO] Time elapsed (h:m:s.ms) {total_time}") 

    
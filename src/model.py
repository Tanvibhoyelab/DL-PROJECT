"""Siamese U-Net matching change_detector.pth."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


class SiameseUNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
    ):

        super().__init__()

        # -------------------------------------------------
        # Shared encoder
        # -------------------------------------------------

        self.e1 = ConvBlock(
            in_channels,
            base_channels,
        )

        self.e2 = ConvBlock(
            base_channels,
            base_channels * 2,
        )

        self.e3 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.e4 = ConvBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

        self.u3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2,
        )

        self.d3 = ConvBlock(
            base_channels * 8,
            base_channels * 4,
        )

        self.u2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )

        self.d2 = ConvBlock(
            base_channels * 4,
            base_channels * 2,
        )

        self.u1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )

        self.d1 = ConvBlock(
            base_channels * 2,
            base_channels,
        )

        # -------------------------------------------------
        # Binary change head
        # -------------------------------------------------

        self.head = nn.Conv2d(
            base_channels,
            1,
            kernel_size=1,
        )

    def encode(
        self,
        x: torch.Tensor,
    ):

        f1 = self.e1(x)

        f2 = self.e2(
            self.pool(f1)
        )

        f3 = self.e3(
            self.pool(f2)
        )

        f4 = self.e4(
            self.pool(f3)
        )

        return f1, f2, f3, f4

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:

        # Shared encoder
        a1, a2, a3, a4 = self.encode(
            before
        )

        b1, b2, b3, b4 = self.encode(
            after
        )

        # Feature differences
        diff1 = torch.abs(
            a1 - b1
        )

        diff2 = torch.abs(
            a2 - b2
        )

        diff3 = torch.abs(
            a3 - b3
        )

        diff4 = torch.abs(
            a4 - b4
        )

        # Decoder
        x = self.u3(
            diff4
        )

        x = self.d3(
            torch.cat(
                [
                    x,
                    diff3,
                ],
                dim=1,
            )
        )

        x = self.u2(
            x
        )

        x = self.d2(
            torch.cat(
                [
                    x,
                    diff2,
                ],
                dim=1,
            )
        )

        x = self.u1(
            x
        )

        x = self.d1(
            torch.cat(
                [
                    x,
                    diff1,
                ],
                dim=1,
            )
        )

        return self.head(x)


class DiceBCELoss(nn.Module):
    """Combined loss:

        L = L_BCE(pos_weight) + L_Dice
        L_Dice = 1 - (2|P ∩ G| + eps) / (|P| + |G| + eps)

    where P is the predicted change probability map (post-sigmoid) and G is the
    ground-truth binary mask. BCE is computed on raw logits (numerically
    stable), Dice on the sigmoid probabilities.

    Change masks are heavily imbalanced (roughly 5% changed pixels in LEVIR-CD),
    so unweighted BCE lets the network collapse towards predicting "no change".
    ``pos_weight`` is passed in by the training script, measured from the
    training labels rather than guessed.
    """

    def __init__(self, eps: float = 1e-6, pos_weight: float | None = None):
        super().__init__()
        self.eps = eps
        weight = None if pos_weight is None else torch.tensor([float(pos_weight)])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        # Sum over channel/height/width, keep batch dim, then average over batch.
        intersection = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + self.eps) / (union + self.eps))
        dice_loss = dice_loss.mean()

        return bce_loss + dice_loss


def build_model(
    in_channels: int = 3,
    base_channels: int = 32,
) -> SiameseUNet:

    return SiameseUNet(
        in_channels=in_channels,
        base_channels=base_channels,
    )


def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
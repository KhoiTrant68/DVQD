import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.stage1.modules.discriminator.model import weights_init
from src.utils.util_modules import instantiate_from_config


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def log(t, eps=1e-10):
    return torch.log(t + eps)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_g_loss(logits_fake):
    return -torch.mean(logits_fake)


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def bce_discr_loss(logits_real, logits_fake):
    return (
        -log(1 - torch.sigmoid(logits_fake)) - log(torch.sigmoid(logits_real))
    ).mean()


def bce_gen_loss(logits_fake):
    return -log(torch.sigmoid(logits_fake)).mean()


class VQLPIPSWithDiscriminator(nn.Module):
    """
    A class that combines VQ, LPIPS, and a discriminator for training.
    """

    def __init__(
        self,
        disc_start,
        disc_config,
        disc_init,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        disc_conditional=False,
        disc_adaptive_loss=True,
        disc_loss="hinge",
        disc_weight_max=None,
        budget_loss_config=None,
    ):
        """
        Initializes the VQLPIPSWithDiscriminator class.

        Parameters:
        - disc_start (int): The step to start using the discriminator.
        - disc_config (dict): Configuration for the discriminator.
        - disc_init (bool): Whether to initialize the discriminator.
        - codebook_weight (float, optional): The weight for the codebook loss. Defaults to 1.0.
        - pixelloss_weight (float, optional): The weight for the pixel loss. Defaults to 1.0.
        - disc_factor (float, optional): The factor for the discriminator loss. Defaults to 1.0.
        - disc_weight (float, optional): The weight for the discriminator. Defaults to 1.0.
        - perceptual_weight (float, optional): The weight for the perceptual loss. Defaults to 1.0.
        - disc_conditional (bool, optional): Whether the discriminator is conditional. Defaults to False.
        - disc_adaptive_loss (bool, optional): Whether to use adaptive loss for the discriminator. Defaults to True.
        - disc_loss (str, optional): The type of discriminator loss. Defaults to "hinge".
        - disc_weight_max (float or None, optional): The maximum weight for the discriminator. Defaults to None.
        - budget_loss_config (dict or None, optional): Configuration for budget loss. Defaults to None.
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "bce"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight

        self.perceptual_loss = lpips.LPIPS(net="vgg").eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator_iter_start = disc_start
        self.discriminator = instantiate_from_config(disc_config)
        if disc_init:
            self.discriminator = self.discriminator.apply(weights_init)

        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
            self.gen_loss = hinge_g_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
            self.gen_loss = hinge_g_loss
        elif disc_loss == "bce":
            self.disc_loss = bce_discr_loss
            self.gen_loss = bce_gen_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.disc_adaptive_loss = disc_adaptive_loss
        self.disc_weight_max = disc_weight_max

        self.budget_loss_config = budget_loss_config
        if budget_loss_config is not None:
            self.budget_loss = instantiate_from_config(budget_loss_config)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Calculates the adaptive weight for the discriminator loss.

        Parameters:
        - nll_loss (Tensor): The negative log-likelihood loss.
        - g_loss (Tensor): The generator loss.
        - last_layer (Tensor or None, optional): The last layer of the model. Defaults to None.

        Returns:
        - Tensor: The adaptive weight for the discriminator loss.
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        gate=None,
    ):
        """
        Forward pass for the VQLPIPSWithDiscriminator model.

        Parameters:
        - codebook_loss (Tensor): The codebook loss.
        - inputs (Tensor): The input data.
        - reconstructions (Tensor): The reconstructed data.
        - optimizer_idx (int): The index of the optimizer (0 for generator, 1 for discriminator).
        - global_step (int): The current global step.
        - last_layer (Tensor or None, optional): The last layer of the model. Defaults to None.
        - cond (Tensor or None, optional): Conditional data for the discriminator. Defaults to None.
        - split (str, optional): The data split (e.g., "train", "val"). Defaults to "train".
        - gate (Tensor or None, optional): The gate for budget loss. Defaults to None.

        Returns:
        - Tuple[Tensor, dict]: The loss and a dictionary of logs.
        """
        # Reconstruction loss (pixel-level)
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        # Perceptual loss using lpips
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        # Generator update (optimizer_idx == 0)
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )
            g_loss = self.gen_loss(logits_fake)

            if self.disc_adaptive_loss:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
                if (
                    self.disc_weight_max is not None
                ):  # Adds the limit on the maximum value of disc_weight
                    d_weight.clamp_max_(self.disc_weight_max)
            else:  # if not adaptive, directly use disc_weight_max as d_weight
                d_weight = torch.tensor(self.disc_weight_max)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                nll_loss
                + d_weight * disc_factor * g_loss
                + self.codebook_weight * codebook_loss.mean()
            )

            if gate is not None and self.budget_loss_config is not None:
                budget_loss = self.budget_loss(gate=gate)
                loss = loss + budget_loss

                log = {
                    "{}_total_loss".format(split): loss.clone().detach().mean(),
                    "{}_quant_loss".format(split): codebook_loss.detach().mean(),
                    "{}_nll_loss".format(split): nll_loss.detach().mean(),
                    "{}_rec_loss".format(split): rec_loss.detach().mean(),
                    "{}_p_loss".format(split): p_loss.detach().mean(),
                    "{}_d_weight".format(split): d_weight.detach(),
                    "{}_disc_factor".format(split): torch.tensor(disc_factor),
                    "{}_g_loss".format(split): g_loss.detach().mean(),
                    "{}_budget_loss".format(split): budget_loss.detach().mean(),
                }
                return loss, log

            log = {
                "{}_total_loss".format(split): loss.clone().detach().mean(),
                "{}_quant_loss".format(split): codebook_loss.detach().mean(),
                "{}_nll_loss".format(split): nll_loss.detach().mean(),
                "{}_rec_loss".format(split): rec_loss.detach().mean(),
                "{}_p_loss".format(split): p_loss.detach().mean(),
                "{}_d_weight".format(split): d_weight.detach(),
                "{}_disc_factor".format(split): torch.tensor(disc_factor),
                "{}_g_loss".format(split): g_loss.detach().mean(),
            }
            return loss, log

        # Discriminator update (optimizer_idx == 1)
        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}_disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}_logits_real".format(split): logits_real.detach().mean(),
                "{}_logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log

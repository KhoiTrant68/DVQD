import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.stage1.modules.discriminator.model import weights_init
from src.utils.util_modules import instantiate_from_config


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    return value if global_step < threshold else weight


def log(t, eps=1e-10):
    return torch.log(t.clamp(min=eps))


def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def hinge_g_loss(logits_fake):
    return -logits_fake.mean()


def vanilla_d_loss(logits_real, logits_fake):
    return 0.5 * (F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean())


def bce_discr_loss(logits_real, logits_fake):
    return (
        -log(1 - torch.sigmoid(logits_fake)) - log(torch.sigmoid(logits_real))
    ).mean()


def bce_gen_loss(logits_fake):
    return -log(torch.sigmoid(logits_fake)).mean()


class VQLPIPSWithDiscriminator(nn.Module):
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
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "bce"]

        self.perceptual_loss = LearnedPerceptualImagePatchSimilarity(
            net_type="squeeze"
        ).eval()

        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight

        self.discriminator_iter_start = disc_start
        self.discriminator = instantiate_from_config(disc_config)
        if disc_init:
            self.discriminator.apply(weights_init)

        loss_functions = {
            "hinge": (hinge_d_loss, hinge_g_loss),
            "vanilla": (vanilla_d_loss, hinge_g_loss),
            "bce": (bce_discr_loss, bce_gen_loss),
        }
        self.disc_loss, self.gen_loss = loss_functions[disc_loss]
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")

        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.disc_adaptive_loss = disc_adaptive_loss
        self.disc_weight_max = disc_weight_max

        self.budget_loss_config = budget_loss_config
        if budget_loss_config is not None:
            self.budget_loss = instantiate_from_config(budget_loss_config)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
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
        inputs = inputs.to(torch.float32)
        reconstructions = reconstructions.to(torch.float32)
        rec_loss = torch.abs(inputs - reconstructions)
        p_loss = (
            self.perceptual_loss(inputs, reconstructions)
            if self.perceptual_weight > 0
            else 0.0
        )
        rec_loss += self.perceptual_weight * p_loss

        nll_loss = rec_loss.mean()

        if optimizer_idx == 0:
            discriminator_input = (
                reconstructions
                if cond is None
                else torch.cat((reconstructions, cond), dim=1)
            )
            logits_fake = self.discriminator(discriminator_input)
            g_loss = self.gen_loss(logits_fake)

            d_weight = (
                self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
                if self.disc_adaptive_loss
                else self.disc_weight_max
            )
            if self.disc_weight_max is not None:
                d_weight = d_weight.clamp_max(self.disc_weight_max)

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
                loss += budget_loss

            log_dict = {
                f"{split}_total_loss": loss.detach().mean(),
                f"{split}_quant_loss": codebook_loss.detach().mean(),
                f"{split}_nll_loss": nll_loss.detach(),
                f"{split}_rec_loss": rec_loss.detach().mean(),
                f"{split}_p_loss": p_loss.detach(),
                f"{split}_d_weight": d_weight.detach(),
                f"{split}_disc_factor": torch.tensor(disc_factor),
                f"{split}_g_loss": g_loss.detach(),
                f"{split}_budget_loss": (
                    budget_loss.detach().mean()
                    if gate is not None and self.budget_loss_config is not None
                    else None
                ),
            }
            return loss, log_dict

        if optimizer_idx == 1:
            discriminator_input_real = (
                inputs if cond is None else torch.cat((inputs, cond), dim=1)
            )
            logits_real = self.discriminator(discriminator_input_real.detach())
            discriminator_input_fake = (
                reconstructions
                if cond is None
                else torch.cat((reconstructions, cond), dim=1)
            )
            logits_fake = self.discriminator(discriminator_input_fake.detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log_dict = {
                f"{split}_disc_loss": d_loss.detach(),
                f"{split}_logits_real": logits_real.detach().mean(),
                f"{split}_logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, log_dict

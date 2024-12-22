import torch
from PIL import Image

from comfy.cli_args import args, LatentPreviewMethod
import comfy.model_management
import comfy.utils
from einops import rearrange

MAX_PREVIEW_RESOLUTION = args.preview_size

def preview_to_image(latent_image):
        latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            ).to(device="cpu", dtype=torch.uint8, non_blocking=comfy.model_management.device_supports_non_blocking(latent_image.device))
        latents = [Image.fromarray(a_frame.numpy()) for a_frame in latents_ubyte]
        return latents

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("GIF", preview_image, MAX_PREVIEW_RESOLUTION)

class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self):
        latent_rgb_factors = [[-0.0395, -0.0331,  0.0445],
                            [ 0.0696,  0.0795,  0.0518],
                            [ 0.0135, -0.0945, -0.0282],
                            [ 0.0108, -0.0250, -0.0765],
                            [-0.0209,  0.0032,  0.0224],
                            [-0.0804, -0.0254, -0.0639],
                            [-0.0991,  0.0271, -0.0669],
                            [-0.0646, -0.0422, -0.0400],
                            [-0.0696, -0.0595, -0.0894],
                            [-0.0799, -0.0208, -0.0375],
                            [ 0.1166,  0.1627,  0.0962],
                            [ 0.1165,  0.0432,  0.0407],
                            [-0.2315, -0.1920, -0.1355],
                            [-0.0270,  0.0401, -0.0821],
                            [-0.0616, -0.0997, -0.0727],
                            [ 0.0249, -0.0469, -0.1703]]
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = torch.tensor([0.0259, -0.0192, -0.0761], device="cpu")

    def decode_latent_to_preview(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)
        # x0 = rearrange(x0, "b c h w -> (b h) w c" )
        x0 = rearrange(x0, "b c h w -> b h w c" ) * 0.6
        latent_image = torch.nn.functional.linear(x0, self.latent_rgb_factors,
                                                    bias=self.latent_rgb_factors_bias)
        return preview_to_image(latent_image)


def get_previewer():
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        # TODO previewer method

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB

        if previewer is None:
            previewer = Latent2RGBPreviewer()
    return previewer

def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG", "GIF"]:
        preview_format = "JPEG"

    previewer = get_previewer()

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback


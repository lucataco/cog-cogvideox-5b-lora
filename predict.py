# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import re
import time
import torch
import requests
import subprocess
from PIL import Image
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download, list_repo_files

MODEL_CACHE = "CogVideoX"
MODEL_URL = "https://weights.replicate.delivery/default/THUDM/CogVideo/model_cache.tar"
IMAGE_CACHE = "CogVideoX-Image"
IMAGE_URL = "https://weights.replicate.delivery/default/THUDM/CogVideo/model_cache_i2v.tar"

# Environment setup
ENV_VARS = {
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HOME": MODEL_CACHE,
    "TORCH_HOME": MODEL_CACHE,
    "HF_DATASETS_CACHE": MODEL_CACHE,
    "TRANSFORMERS_CACHE": MODEL_CACHE,
    "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
}
os.environ.update(ENV_VARS)

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.last_loaded_lora = None

        print("Loading CogVideoX weights")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.pipe_text = CogVideoXPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.pipe_text.enable_model_cpu_offload()
        self.pipe_text.vae.enable_tiling()
        
        print("setup took: ", time.time() - start)

    def load_lora_weights(self, pipe, hf_lora, lora_scale):
        """Helper function to load LoRA weights based on input type"""
        if self.last_loaded_lora is not None:
            pipe.unload_lora_weights()

        try:
            if re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$", hf_lora):
                print(f"Downloading LoRA weights from HF path: {hf_lora}")
                # List files in the repo to find the safetensors file
                files = list_repo_files(hf_lora)
                safetensors_files = [f for f in files if f.endswith('.safetensors')]
                
                if not safetensors_files:
                    raise ValueError(f"No .safetensors file found in the repository: {hf_lora}")
                
                # Use the first .safetensors file found
                weight_name = safetensors_files[0]
                local_file = hf_hub_download(hf_lora, filename=weight_name)
                pipe.load_lora_weights(local_file)
            elif hf_lora.startswith("https://huggingface.co"):
                print(f"Downloading LoRA weights from HF URL: {hf_lora}")
                huggingface_slug = re.search(r"^https?://huggingface.co/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)", hf_lora).group(1)
                weight_name = hf_lora.split('/')[-1]
                local_file = hf_hub_download(huggingface_slug, filename=weight_name)
                pipe.load_lora_weights(local_file)
            elif hf_lora.endswith('.safetensors'):
                print(f"Downloading LoRA weights from URL: {hf_lora}")
                local_file = "/tmp/lora_weights.safetensors"
                response = requests.get(hf_lora)
                response.raise_for_status()
                with open(local_file, 'wb') as f:
                    f.write(response.content)
                pipe.load_lora_weights(local_file)
            else:
                raise ValueError(f"Unsupported LoRA input: {hf_lora}")

            pipe.set_adapters(["cogvideox-lora"], adapter_weights=lora_scale)
            self.last_loaded_lora = hf_lora
            print(f"Successfully loaded LoRA weights from: {hf_lora}")
        except Exception as e:
            print(f"Error loading LoRA weights: {str(e)}")
            print("Continuing without LoRA weights...")
            self.last_loaded_lora = None


    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest",
        ),
        image: Path = Input(
            description="Input image. Will be resized to 720x480", default=None
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=6
        ),
        num_frames: int = Input(
            description="Number of frames for the output video", default=49
        ),
        hf_lora: str = Input(
            description="Huggingface path or URL to a LoRA. Ex: a-r-r-o-w/cogvideox-disney-adamw-4000-0.0003-constant", default=None,
        ),
        lora_scale: float = Input(
            description="Scale for the LoRA weights",
            ge=0,le=1, default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator(device="cuda").manual_seed(seed)
        pipe = self.pipe_text

        if hf_lora:
            if self.last_loaded_lora != hf_lora:
                print("New LoRA weights detected")
                self.load_lora_weights(pipe, hf_lora, lora_scale)
            else:
                print("Same LoRA weights already loaded")
                pipe.set_adapters(["cogvideox-lora"], adapter_weights=lora_scale)
        else:
            print("No LoRA weights detected")

        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

        out_path = "/tmp/output.mp4"
        export_to_video(video, out_path, fps=8)
        return Path(out_path)

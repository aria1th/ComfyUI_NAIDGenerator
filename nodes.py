import random
try:
    import dotenv
except ImportError:
    print("dotenv not installed, manual login required")
import base64
from hashlib import blake2b
import argon2

import requests
import json
import time

from os import environ as env
import zipfile
import io
from pathlib import Path
import folder_paths
from datetime import datetime, timedelta
import torch
import comfy.utils
import math
import numpy as np
from PIL import Image, ImageOps

from .wildcards_nai import NAITextWildcards

# cherry-picked from novelai_api.utils
def argon_hash(email: str, password: str, size: int, domain: str) -> str:
    pre_salt = f"{password[:6]}{email}{domain}"
    blake = blake2b(digest_size=16)
    blake.update(pre_salt.encode())
    salt = blake.digest()
    raw = argon2.low_level.hash_secret_raw(password.encode(), salt, 2, int(2000000 / 1024), 1, size, argon2.low_level.Type.ID,)
    hashed = base64.urlsafe_b64encode(raw).decode()
    return hashed

def get_access_key(email: str, password: str) -> str:
    return argon_hash(email, password, 64, "novelai_data_access_key")[:64]


BASE_URL="https://api.novelai.net"
def login(key) -> str:
    response = requests.post(f"{BASE_URL}/user/login", json={ "key": key })
    response.raise_for_status()
    return response.json()["accessToken"]

def generate_image(access_token, prompt, model, action, parameters:dict, timeout=120.0):
    data = { "input": prompt, "model": model, "action": action, "parameters": parameters }
    response = requests.post(f"{BASE_URL}/ai/generate-image", json=data, headers={ "Authorization": f"Bearer {access_token}" }, timeout=timeout)
    # 429 Too Many Requests then raise RuntimeError
    if response.status_code == 429:
        errors.log_and_print("429 Error: Too Many Requests (rate limit exceeded)")
        raise RuntimeError("429 Error: Too Many Requests (rate limit exceeded)")
    response.raise_for_status()
    return response.content


def imageToBase64(image):
    i = 255. * image[0].cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    image_bytesIO = io.BytesIO()
    img.save(image_bytesIO, format="png")
    return base64.b64encode(image_bytesIO.getvalue()).decode()

def naimaskToBase64(image):
    i = 255. * image[0].cpu().numpy()
    i = np.clip(i, 0, 255).astype(np.uint8)
    alpha = np.sum(i, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    rgba = np.dstack((i, alpha))
    img = Image.fromarray(rgba)
    image_bytesIO = io.BytesIO()
    img.save(image_bytesIO, format="png")
    return base64.b64encode(image_bytesIO.getvalue()).decode()


class ImageToNAIMask:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image": ("IMAGE",) } }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/utils"
    def convert(self, image):
        samples = image.movedim(-1,1)
        width = math.ceil(samples.shape[3] / 64) * 8
        height = math.ceil(samples.shape[2] / 64) * 8
        s = comfy.utils.common_upscale(samples, width, height, "nearest-exact", "disabled")
        s = s.movedim(1,-1)
        naimaskToBase64(s)
        return (s,)

class ModelOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["safe-diffusion", "nai-diffusion", "nai-diffusion-furry", "nai-diffusion-2", "nai-diffusion-3"], { "default": "nai-diffusion-3" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, model, option=None):
        option = option or {}
        option["model"] = model
        return (option,)

class Img2ImgOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", { "default": 0.70, "min": 0.01, "max": 0.99, "step": 0.01, "display": "number" }),
                "noise": ("FLOAT", { "default": 0.00, "min": 0.00, "max": 0.99, "step": 0.02, "display": "number" }),
            },
#            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, strength, noise, option=None):
        option = option or {}
        option["img2img"] = (image, strength, noise)
        return (option,)

class InpaintingOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "add_original_image": ("BOOLEAN", { "default": True }),
            },
#            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, mask, add_original_image, option=None):
        option = option or {}
        option["infill"] = (image, mask, add_original_image)
        return (option,)

class BiasedRandom:
    """
    Generates a random number with bias
    Bigger numbers have smaller chance of being generated
    """
    def __init__(self, min_val, max_val, bias = 0.02, chance=0.8):
        self.min = min_val
        self.max = max_val
        self.bias = bias
        self.chance = chance
        self.max_depth = 20
    def generate(self, min_val=None,max_val=None, depth=0):
        if min_val is None:
            min_val = self.min
        if max_val is None:
            max_val = self.max
        # roll a dice to select area
        if random.random() < self.chance or depth >= self.max_depth:
            # small area
            min_val = min_val
            max_val = min_val + (max_val - min_val) * self.bias
            return random.randint(int(min_val), int(max_val))
        else:
            # recusive call
            new_min = min_val + (max_val - min_val) * self.bias
            new_max = max_val
            return self.generate(new_min, new_max, depth+1)

class ErrorStatistics:
    """
    Logs errors and calculates statistics
    """
    def __init__(self) -> None:
        from collections import defaultdict
        self.logged_reasons = defaultdict(int)
    def log(self, reason):
        self.logged_reasons[reason] += 1
    def get_statistics(self):
        # pretty print
        print("Error Statistics:")
        for reason, count in self.logged_reasons.items():
            print(f"{reason}: {count}, {count / sum(self.logged_reasons.values()) * 100:.2f}%")
        print(f"Total: {sum(self.logged_reasons.values())}")
    def log_and_print(self, reason):
        # log the reason and print the statistics
        self.log(reason)
        self.get_statistics()

def truncate_tokens(prompt, max_tokens=225):
    """
    Truncates the prompt to max_tokens
    """
    tokens = prompt.split(",")
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    return prompt

class NAISmeaRandom:
    """
    Selects a random Smea option
    """
    def __init__(self):
        self.options = {
            "none" : ["none"],
            "SMEA" : ["none", "SMEA"],
            "SMEA+DYN" : ["none", "SMEA", "SMEA+DYN"],
        }
    def generate(self, pools="SMEA+DYN", seed=0):
        instance = random.Random(seed)
        choice = instance.choice(self.options[pools])
        print(f"Selected {choice} from {self.options[pools]}")
        return (choice,)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed" : ("INT", { "default": 0, "min": 0, "max": 9999999999, "step": 1, "display": "number" }),
            },
            "optional": { "pools": (["none", "SMEA", "SMEA+DYN"], { "default": "SMEA+DYN" }) ,
                         },
        }
    RETURN_TYPES = (["none", "SMEA", "SMEA+DYN"],)
    FUNCTION = "generate"
    CATEGORY = "NovelAI"

class UniformRandomFloat:
    """
    Selects a random float from min to max
    Fallbacks to default if min is greater than max
    """
    def __init__(self):
        pass
    def generate(self, min_val, max_val, decimal_places, seed=0):
        if min_val > max_val:
            return min_val
        instance = random.Random(seed)
        value = instance.uniform(min_val, max_val)
        # prune to decimal places - 0 = int, 1 = 1 decimal place,...
        value = round(value, decimal_places)
        print(f"Selected {value} from {min_val} to {max_val}")
        return (value,)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_val": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.02, "display": "number" }),
                "max_val": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.02, "display": "number" }),
                "decimal_places": ("INT", { "default": 1, "min": 0, "max": 10, "step": 1, "display": "number" }),
                "seed" : ("INT", { "default": 0, "min": 0, "max": 9999999999, "step": 1, "display": "number" }),
            },
        }
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "generate"
    CATEGORY = "NovelAI"

errors = ErrorStatistics()
class GenerateNAID:
    def __init__(self):
        try:
            dotenv.load_dotenv()
        except Exception as e:
            print(f"dotenv not loaded: {e}")
        self.logged_username = None
        self.logged_password = None
        self.access_token_fixed = None
        self.access_token = None
        self.output_dir = folder_paths.get_output_directory()
        self.run_started = None
        self.initial_run_started= None
        self.total_created = 0
        self.total_all_created = 0
    
    def handle_login(self):
        """
        Logins and returns an access token
        """
        if self.access_token_fixed is not None:
            return self.access_token_fixed
        if "NAI_ACCESS_KEY" in env:
            access_key = env["NAI_ACCESS_KEY"]
        elif "NAI_USERNAME" in env and "NAI_PASSWORD" in env:
            username = env["NAI_USERNAME"]
            password = env["NAI_PASSWORD"]
            access_key = get_access_key(username, password)
        elif self.logged_username and self.logged_password:
            username = self.logged_username
            password = self.logged_password
            access_key = get_access_key(username, password)
        elif "NAI_ACCESS_TOKEN" in env:
            access_token = env["NAI_ACCESS_TOKEN"]
            self.access_token = access_token
            return access_token
        else:
            raise RuntimeError("Please ensure that NAI_ACCESS_KEY or NAI_USERNAME and NAI_PASSWORD are set in your environment")
        access_token = login(access_key)
        self.access_token = access_token
        return access_token

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "size": (["Portrait", "Landscape", "Square", "Random", "Custom(Paid)", "Large Portrait(Paid)", "Large Landscape(Paid)"], { "default": "Portrait" }),
                "width": ("INT", { "default": 832, "min": 64, "max": 1600, "step": 64, "display": "number" }),
                "height": ("INT", { "default": 1216, "min": 64, "max": 1600, "step": 64, "display": "number" }),
                "positive": ("STRING", { "default": "{}, best quality, amazing quality, very aesthetic, absurdres", "multiline": True, "dynamicPrompts": False }),
                "negative": ("STRING", { "default": "lowres", "multiline": True, "dynamicPrompts": False }),
                "steps": ("INT", { "default": 28, "min": 0, "max": 50, "step": 1, "display": "number" }),
                "cfg": ("FLOAT", { "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "number" }),
                "smea": (["none", "SMEA", "SMEA+DYN"], { "default": "none" }),
                "sampler": (["k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral", "k_dpmpp_2m", "k_dpmpp_sde", "ddim"], { "default": "k_euler" }),
                "scheduler": (["native", "karras", "exponential", "polyexponential"], { "default": "native" }),
                "seed": ("INT", { "default": 0, "min": 0, "max": 9999999999, "step": 1, "display": "number" }),
                "uncond_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05, "display": "number" }),
                "cfg_rescale": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02, "display": "number" }),
                "delay_max": ("FLOAT", { "default": 2.1, "min": 2.0, "max": 24000.0, "step": 0.1, "display": "number" }), # delay in seconds
                "fallback_black": ("INT", { "default": 1, "min": 0, "max": 1, "step": 1, "display": "number" }), # 0: no fallback, 1: fallback to black image
                "delay_min": ("FLOAT", { "default": 2.0, "min": 0.0, "max": 24000.0, "step": 0.1, "display": "number" }), # delay in seconds
            },
            "optional": { "option": ("NAID_OPTION",) ,
                            "username": ("STRING", { "default": "", "multiline": False, "dynamicPrompts": False }),
                            "password": ("STRING", { "default": "", "multiline": False, "dynamicPrompts": False }),
                            "runtime_limit_min" : ("INT", { "default": 0, "min": 0, "max": 24000, "step": 1, "display": "number" }), # runtime limit in seconds
                            "runtime_limit_max" : ("INT", { "default": 0, "min": 0, "max": 24000, "step": 1, "display": "number" }), # runtime limit in seconds
                            "sleep_min" : ("INT", { "default": 0, "min": 0, "max": 24000, "step": 1, "display": "number" }), # sleep time in seconds
                            "sleep_max" : ("INT", { "default": 0, "min": 0, "max": 24000, "step": 1, "display": "number" }), # sleep time in seconds
                            "save" : (["True", "False"], {"default": "False"}), # save image to comfy output dir
                            "nai_token" : ("STRING", { "default": "", "multiline": False, "dynamicPrompts": False }), # override access token
                         },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "NovelAI"
    
    def sanitize_options(self, size, width, height, steps, option, positive, negative):
        """
        Validates the free options
        """
        if "Paid" in size:
            # sanitize Large Portrait and Large Landscape
            if "Large Portrait" in size:
                width = 1024
                height = 1536
            elif "Large Landscape" in size:
                width = 1536
                height = 1024
            return size, width, height, steps, option, positive, negative
        # if width or height is not default, warn and reset to default
        # truncate positive -> 225, negative -> 225
        positive_pruned, negative_pruned = truncate_tokens(positive), truncate_tokens(negative)
        # if different, print warning
        if positive != positive_pruned:
            print(f"Overriding positive prompt to {positive_pruned}")
            positive = positive_pruned
        if negative != negative_pruned:
            print(f"Overriding negative prompt to {negative_pruned}")
            negative = negative_pruned
        if size == 'Random':
            size = random.choice(["Portrait", "Landscape", "Square"])
        if size == "Portrait":
            if width != 832 or height != 1216:
                print("Overriding width and height to default values for Portrait")
                width = 832
                height = 1216 
        elif size == "Landscape":
            if width != 1216 or height != 832:
                print("Overriding width and height to default values for Landscape")
                width = 1216
                height = 832
        elif size == "Square":
            if width != 1024 or height != 1024:
                print("Overriding width and height to default values for Square")
                width = 1024
                height = 1024
        if steps > 28:
            print("Overriding steps to 28")
            steps = 28
        if option:
            if "img2img" in option or "infill" in option:
                print("Overriding option to None")
                option = {}
        return size, width, height, steps, option, positive, negative

    def generate(self, size, width, height, positive, negative, steps, cfg, smea, sampler, scheduler, seed, uncond_scale,
                 cfg_rescale, delay_max, fallback_black, delay_min, option=None, username=None, password=None,runtime_limit_min=0, runtime_limit_max=0, sleep_min=0, sleep_max=0, save=False,
                 nai_token=None):
        save = save == "True"
        # ref. novelai_api.ImagePreset
        # We override the default values here for non-custom sizes
        size, width, height, steps, option, positive, negative = self.sanitize_options(size, width, height, steps, option, positive, negative)
        assert cfg > 0, "cfg must be greater than 0"
        if username and password:
            self.username = username
            self.password = password
        if nai_token:
            self.access_token_fixed = nai_token # override access token
            self.access_token = nai_token
        if self.run_started is None:
            self.run_started = datetime.now()
            self.initial_run_started = datetime.now()
        if runtime_limit_min > 0:
            runtime = (datetime.now() - self.run_started).total_seconds()
            if runtime_limit_max <= runtime_limit_min:
                runtime_limit_max = runtime_limit_min + 1
            runtime_limit = random.randint(runtime_limit_min, runtime_limit_max)
            if sleep_min == 0 and sleep_max == 0:
                sleep_time = 0
            elif sleep_min > 0 and sleep_max > 0:
                s_min, s_max = min(sleep_min, sleep_max), max(sleep_min, sleep_max)
                sleep_time = random.randint(s_min, s_max)
            else:
                raise RuntimeError("Invalid sleep_min and sleep_max")
            if sleep_time == 0 and runtime > runtime_limit:
                raise RuntimeError("Runtime limit exceeded, won't sleep but raised exception")
            elif sleep_time > 0 and runtime > runtime_limit:
                print(f"Runtime limit exceeded, sleeping for {sleep_time} seconds")
                restart_at = datetime.now() + timedelta(seconds=sleep_time)
                print(f"Restarts at {restart_at}")
                time.sleep(sleep_time)
                self.run_started = datetime.now()
                self.total_created = 0
                runtime = 0
            total_runtime = (datetime.now() - self.initial_run_started).total_seconds()
            print(f"Running for {runtime} seconds")
            print(f"Total runtime {total_runtime} seconds")
            print(f"Created {self.total_created} images at {runtime / max(self.total_created,1)} seconds per image")
            print(f"Created {self.total_all_created} images in total at {total_runtime / max(self.total_all_created,1)} seconds per image")
            # bugs statistics
            errors.get_statistics()
        params = {
            "legacy": False,
            "quality_toggle": False,
            "width": width,
            "height": height,
            "n_samples": 1,
            "seed": seed,
            "extra_noise_seed": seed,
            "sampler": sampler,
            "steps": steps,
            "scale": cfg,
            "uncond_scale": uncond_scale,
            "negative_prompt": negative,
            "sm": (smea == "SMEA" or smea == "SMEA+DYN") and sampler != "ddim",
            "sm_dyn": smea == "SMEA+DYN" and sampler != "ddim",
            "decrisper": False,
            "controlnet_strength": 1.0,
            "add_original_image": False,
            "cfg_rescale": cfg_rescale,
            "noise_schedule": scheduler,
        }

        model = "nai-diffusion-3"
        action = "generate"
        if "Paid" in size:
            if option:
                if "img2img" in option:
                    action = "img2img"
                    image, strength, noise = option["img2img"]
                    params["image"] = imageToBase64(image)
                    params["strength"] = strength
                    params["noise"] = noise
                elif "infill" in option:
                    action = "infill"
                    image, mask, add_original_image = option["infill"]
                    params["image"] = imageToBase64(image)
                    params["mask"] = naimaskToBase64(mask)
                    params["add_original_image"] = add_original_image

                if "model" in option:
                    model = option["model"]

        if action == "infill" and model != "nai-diffusion-2":
            model = f"{model}-inpainting"

        retry_max = 5
        retry_count = 0
        zipped_bytes = None
        while retry_count < retry_max:
            try:
                zipped_bytes = generate_image(self.access_token, positive, model, "generate", params, timeout = delay_max + 60)
                break
            # handle timeout, 500 errors, SSL errors
            except Exception as e:
                # log HTTPError status codes
                if isinstance(e, requests.exceptions.HTTPError):
                    errors.log_and_print(f"HTTPError: {e.response.status_code}")
                elif isinstance(e, requests.exceptions.SSLError):
                    errors.log_and_print(f"SSLError: {e}")
                elif isinstance(e, requests.exceptions.ConnectionError):
                    errors.log_and_print(f"ConnectionError: {e}")
                else:
                    errors.log_and_print(f"Error: {e}")
                # check 500 Internal Server Error, 500 then sleep 5 seconds else sleep 60 seconds (or more)
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 500:
                    print(f"retrying {retry_count} after 5 seconds...")
                    time.sleep(5)
                else:
                    if not isinstance(e, requests.exceptions.Timeout):
                        # wait for 60 seconds
                        print(f"Error: {e}")
                        print(f"retrying {retry_count} after 60 seconds...")
                        time.sleep(60) # sleep for 60 seconds
                    retry_count += 1
                    print(f"Error: {e}")
                    print(f"retrying {retry_count} after 60 seconds...")
                    time.sleep(60) # sleep for 60 seconds
                while True:
                    try:
                        self.handle_login() # refresh access token
                        break
                    except Exception as e:
                        if isinstance(e, requests.exceptions.HTTPError):
                            errors.log_and_print(f"HTTPError: {e.response.status_code}")
                        elif isinstance(e, requests.exceptions.SSLError):
                            errors.log_and_print(f"SSLError: {e}")
                        elif isinstance(e, requests.exceptions.ConnectionError):
                            errors.log_and_print(f"ConnectionError: {e}")
                        else:
                            errors.log_and_print(f"Error: {e}")
                        print(f"retrying {retry_count} after 120-240 seconds, relogin...")
                        time.sleep(random.randint(120, 240)) # sleep for 120-240 seconds
        if zipped_bytes is None and not fallback_black:
            raise RuntimeError("Failed to generate image, possibly due to timeout")
        
        try:
            zipped = zipfile.ZipFile(io.BytesIO(zipped_bytes))
            image_bytes = zipped.read(zipped.infolist()[0]) # only support one n_samples
            i = Image.open(io.BytesIO(image_bytes))
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
        except Exception as exception:
            if fallback_black:
                image = torch.zeros((1, 3, height, width))
            else:
                raise exception
        self.total_created += 1
        self.total_all_created += 1
        if save:
            ## save original png to comfy output dir
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("NAI_autosave", self.output_dir)
            file = f"{filename}_{counter:05}_.png"
            d = Path(full_output_folder)
            d.mkdir(exist_ok=True)
            (d / file).write_bytes(image_bytes)
        biased_random = BiasedRandom(max(delay_min, 2.0), max(delay_max, 2.0), 0.02, 0.8).generate()
        print(f"sleeping for {biased_random} seconds")
        time.sleep(biased_random) # sleep for 2 seconds minimum

        return (image,)


NODE_CLASS_MAPPINGS = {
    "GenerateNAID": GenerateNAID,
    "ModelOptionNAID": ModelOption,
    "Img2ImgOptionNAID": Img2ImgOption,
    "InpaintingOptionNAID": InpaintingOption,
    "ImageToNAIMask": ImageToNAIMask,
    "NAITextWildcards": NAITextWildcards, # Wildcards
    "NAISmeaRandom": NAISmeaRandom,
    "UniformRandomFloat": UniformRandomFloat,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateNAID": "Generate ✒️🅝🅐🅘",
    "ModelOptionNAID": "ModelOption ✒️🅝🅐🅘",
    "Img2ImgOptionNAID": "Img2ImgOption ✒️🅝🅐🅘",
    "InpaintingOptionNAID": "InpaintingOption ✒️🅝🅐🅘",
    "ImageToNAIMask": "Convert Image to NAI Mask",
    "NAITextWildcards": "NAITextWildcards ✒️🅝🅐🅘",
    "NAISmeaRandom": "NAISmeaRandom ✒️🅝🅐🅘",
    "UniformRandomFloat": "UniformRandomFloat",
}
import base64
import concurrent.futures
import gc
import io
import math
import os
import threading
import time
from collections import deque
from typing import Callable, List

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from colorama import Fore, init
from google import genai
from google.genai import types
from openai import OpenAI, OpenAIError
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from vllm.outputs import RequestOutput

init(autoreset=True)

class MinuteRateLimiter:
    def __init__(self, max_calls_per_minute: int):
        self.max_calls = max_calls_per_minute
        self.lock = threading.Lock()
        self.calls = deque()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()

                while self.calls and now - self.calls[0] > 60:
                    self.calls.popleft()

                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return

                wait_time = 60 - (now - self.calls[0])

            time.sleep(max(wait_time, 0.1))

class UniversalGenParams:
    def __init__(
        self,
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 1,
        seed: int = 0,
        stop: List[str] = None,
        reasoning: str = "low",
        thinking_tokens: int = 0,
    ):
        self.n = n
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.stop = stop
        self.reasoning = reasoning
        self.thinking_tokens = thinking_tokens
    
    def __dict__(self):
        return {
            "n": self.n,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "reasoning": self.reasoning,
        }
    
    def get_hf_params(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else None,
            "temperature": self.temperature,
            "do_sample": True if self.temperature > 0 else False,
            "eos_token_id": self.stop if self.stop else None,
        }
    
    def get_vllm_params(self):
        return SamplingParams(
            n=self.n,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            logprobs=None,
            prompt_logprobs=None,
            stop=self.stop,
        )
    
    def get_openai_params(self):
        return {
            "n": self.n,
            "max_completion_tokens": self.max_new_tokens,
            "seed": self.seed,
            "reasoning": self.reasoning,
        }
    
    def get_gemini_params(self):
        return {
            "max_output_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop_sequences": self.stop,
            "candidate_count": self.n,
        }


class GenerationArgs:
    def __init__(
        self,
        engine_input = None,
        gen_params: UniversalGenParams = UniversalGenParams(),
        response_format: dict = None,
        is_multi_turn_input: bool = False,
        is_batch_input: bool = False,
        use_system_prompt: bool = False,
        add_generation_prompt: bool = True,
        user_role: str = "user",
        model_role: str = "assistant",
        system_role: str = "system",
        chat_prompt_processor: Callable = None,
        chat_prompt_processor_kwargs: dict = {},
    ):
        self.engine_input = engine_input
        self.gen_params = gen_params
        self.response_format = response_format
        self.is_multi_turn_input = is_multi_turn_input
        self.is_batch_input = is_batch_input
        self.use_system_prompt = use_system_prompt
        self.add_generation_prompt = add_generation_prompt
        self.user_role = user_role
        self.model_role = model_role
        self.system_role = system_role
        self.chat_prompt_processor = chat_prompt_processor
        self.chat_prompt_processor_kwargs = chat_prompt_processor_kwargs
    
    def __dict__(self):
        return {
            "engine_input": self.engine_input,
            "gen_params": dict(self.gen_params),
            "response_format": self.response_format,
            "is_multi_turn_input": self.is_multi_turn_input,
            "is_batch_input": self.is_batch_input,
            "use_system_prompt": self.use_system_prompt,
            "add_generation_prompt": self.add_generation_prompt,
            "user_role": self.user_role,
            "model_role": self.model_role,
            "system_role": self.system_role,
            "chat_prompt_processor_kwargs": self.chat_prompt_processor_kwargs,
        }

class VLMInferenceOutput:
   def __init__(
           self,
           output_seqs=None,
           output_objects=None,
           input_prompt=None,
           logits=None,
           finish_reasons=None,
           logprobs=None,
           prompt_logprobs=None,
           cumulative_output_logprobs=None,
           latency=None,
           usage=None,
   ):
       self.input_prompt = input_prompt

       assert isinstance(output_seqs, list), "output_seqs should be a list of sequences."
       self.output_seqs = output_seqs
       self.output_objects = output_objects
       self.logits = logits
       self.finish_reasons = finish_reasons
       self.logprobs = logprobs
       self.prompt_logprobs = prompt_logprobs
       self.cumulative_output_logprobs = cumulative_output_logprobs
       self.latency = latency
       self.usage = usage

    
class VLMInferenceEngine:
    def __init__(self, model_id, backend, backend_kwargs={}):
        self.model_id = model_id
        self.backend = backend
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._image_cache = {}
        self._image_b64_cache = {}
        self.is_internvl = False
        self._load_model(backend_kwargs)

    def _load_model(self, backend_kwargs):
        if self.backend == "vllm":
           self.model = LLM(self.model_id, trust_remote_code=True, **backend_kwargs)
        elif self.backend == "hf":
            print(Fore.CYAN + f"[INFO] Loading HF model: {self.model_id}")
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            model_type = getattr(config, "model_type", "").lower()
            
            self.is_internvl = "internvl" in model_type or "internvl" in self.model_id.lower()
            
            if self.is_internvl and torch.cuda.device_count() > 1:
                print(Fore.YELLOW + "[INFO] Multi-GPU detected. Applying InternVL split_model logic.")
                backend_kwargs["device_map"] = self._get_internvl_device_map()
                backend_kwargs.setdefault("use_flash_attn", True)
            else:
                backend_kwargs.setdefault("device_map", "auto")
                backend_kwargs.setdefault("attn_implementation", "sdpa")

            if self.is_internvl:
                loader_class = AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)
            else:
                loader_class = AutoModelForVision2Seq
                max_pixels = 1280 * 28 * 28
                min_pixels = 256 * 28 * 28
                self.processor = AutoProcessor.from_pretrained(self.model_id, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True)


            self.model = loader_class.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                **backend_kwargs
            ).eval()

        elif self.backend == "openai":
            self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.num_openai_workers = 64
        elif self.backend == "gemini":
            self.model = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini_rate_limiter = MinuteRateLimiter(max_calls_per_minute=100)
        elif self.backend == "vllm-openai":
            base_url = backend_kwargs.get("base_url")
            if base_url is None:
                base_url = "http://localhost:8000/v1"
            print(Fore.YELLOW + f"[INFO] Using OpenAI API at {base_url}")
            self.model = OpenAI(base_url=base_url, api_key=self.model_id, timeout=None)
            self.num_openai_workers = 10
        else:
            raise ValueError("Invalid engine")
    
    def _get_internvl_device_map(self):
        world_size = torch.cuda.device_count()
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        llm_cfg = getattr(config, "llm_config", config)
        num_layers = getattr(llm_cfg, "num_hidden_layers", 32)
        
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu_list = [num_layers_per_gpu] * world_size
        num_layers_per_gpu_list[0] = math.ceil(num_layers_per_gpu_list[0] * 0.5)
        
        device_map = {}
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu_list):
            for j in range(num_layer):
                if layer_cnt < num_layers:
                    device_map[f'language_model.model.layers.{layer_cnt}'] = i
                    layer_cnt += 1
        device_map.update({
            'vision_model': 0, 'mlp1': 0, 'language_model.model.tok_embeddings': 0,
            'language_model.model.embed_tokens': 0, 'language_model.output': 0,
            'language_model.model.norm': 0, 'language_model.lm_head': 0,
            f'language_model.model.layers.{num_layers - 1}': 0
        })
        return device_map
    
    def generate(self, gen_args: GenerationArgs) -> List[VLMInferenceOutput]:
        if not gen_args.is_multi_turn_input and gen_args.use_system_prompt:
            print(Fore.YELLOW + "[WARNING] use_system_prompt is True but is_multi_turn_input is False. Forcing is_multi_turn_input to True.")
            gen_args.is_multi_turn_input = True
        
        if self.backend == "hf":
            return self._hf_generate(gen_args)
        
        if self.backend=="vllm":
            return self._vllm_generate(gen_args)
        elif self.backend=="vllm-openai":
            return self._vllm_openai_generate(gen_args)
        elif self.backend=="openai":
            return self._openai_generate(gen_args)
        elif self.backend == "gemini":
            return self._gemini_generate(gen_args)
        else:
            raise ValueError("Invalid engine")
    
    def _hf_generate(self, gen_args: GenerationArgs) -> List[VLMInferenceOutput]:
        return self._internvl_generate(gen_args) if self.is_internvl else self._hf_standard_generate(gen_args)
    
    def _hf_standard_generate(self, gen_args: GenerationArgs) -> List[VLMInferenceOutput]:
        inputs_list = self._hf_input_preprocessor(gen_args)
        gen_params = gen_args.gen_params.get_hf_params()
        outputs = []
        for inputs in tqdm(inputs_list, desc="HF Standard Inference", disable=len(inputs_list) <= 1):
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **gen_params)
            gen_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
            output_text = self.processor.batch_decode(gen_tokens, skip_special_tokens=True)
            outputs.append(VLMInferenceOutput(output_seqs=output_text))
        return outputs

    def _vllm_generate(self, gen_args: GenerationArgs) -> List[VLMInferenceOutput]:
        prompts = self._vllm_openai_input_preprocessor(gen_args)
        gen_params = gen_args.gen_params.get_vllm_params()
        model_outputs = self._vllm_inference(prompts, gen_args=gen_args, gen_params=gen_params)
        assert len(model_outputs) == len(prompts), "Number of inputs and outputs should be the same."
        outputs = self._vllm_parse_output(model_outputs)
        return outputs
    
    def _vllm_openai_generate(self, gen_args: GenerationArgs) -> List[VLMInferenceOutput]:
        prompts = self._vllm_openai_input_preprocessor(gen_args)
        gen_params = gen_args.gen_params.get_vllm_params()
        
        openai_params = {
            "max_tokens": gen_args.gen_params.max_new_tokens,
            "temperature": gen_args.gen_params.temperature,
            "top_p": gen_args.gen_params.top_p,
            "n": gen_args.gen_params.n,
            "seed": gen_args.gen_params.seed,
        }

        if gen_args.is_batch_input:
            model_outputs = self._vllm_openai_parallel_inference(prompts, openai_params, gen_args.response_format)
            outputs = self._vllm_openai_parse_output(model_outputs, prompts)
        else:
            model_output = self._vllm_openai_inference(prompts[0], openai_params, gen_args.response_format)
            outputs = self._vllm_openai_parse_output([model_output], prompts)
        
        return outputs

    def _openai_generate(self, gen_args: GenerationArgs) -> List[VLMInferenceOutput]:
        prompts = self._vllm_openai_input_preprocessor(gen_args)
        gen_params = gen_args.gen_params.get_openai_params()
        if gen_args.is_batch_input:
            model_outputs = self._openai_parallel_inference(prompts, gen_params=gen_params, response_format=gen_args.response_format)
            outputs = self._openai_parse_output(model_outputs, prompts)
        else:
            model_outputs = self._openai_inference(prompts[0], gen_params=gen_params, response_format=gen_args.response_format)
            outputs = self._openai_parse_output(model_outputs, prompts[0])
        return outputs
    
    def _gemini_generate(self, gen_args: GenerationArgs):
        gen_params = gen_args.gen_params.get_gemini_params()

        if gen_args.is_batch_input:
            contents_list = [self._construct_gemini_chat(x) for x in gen_args.engine_input]
            responses = self.gemini_parallel_inference(contents_list, gen_params)
            
            outputs = []
            for resp, cont in zip(responses, contents_list):
                outputs.extend(self.gemini_parse_output(resp, cont))
            return outputs
        else:
            contents = self._construct_gemini_chat(gen_args.engine_input)
            resp = self.gemini_inference(contents, gen_params)
            return self.gemini_parse_output(resp, contents)

    def _internvl_generate(self, gen_args: GenerationArgs) -> List[VLMInferenceOutput]:
        gen_params = gen_args.gen_params.get_hf_params()
        engine_inputs = gen_args.engine_input if gen_args.is_batch_input else [gen_args.engine_input]
        results = []
        for item in tqdm(engine_inputs, desc="InternVL Inference", disable=not gen_args.is_batch_input and len(engine_inputs) <= 1):
            parts = item if isinstance(item, list) else [item]
            image_input = next((part for part in parts if self._ensure_pil_image(part) is not None), None)
            pixel_values, num_patches = self._internvl_preprocess_image(image_input)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.model.dtype).to(self.device)
            question = next(
                (part for part in reversed(parts) if isinstance(part, str) and not os.path.exists(part)),
                str(parts[-1]),
            )
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question

            start_time = time.time()
            response = self.model.chat(self.tokenizer, pixel_values, question, gen_params,
                                       num_patches_list=[num_patches] if num_patches else None)
            results.append(VLMInferenceOutput(output_seqs=[response], latency=time.time() - start_time))
        return results
    
    def _ensure_pil_image(self, p):
        if isinstance(p, Image.Image):
            return p if self.is_internvl else self._resize_image_optimally(p)
        if isinstance(p, str) and os.path.exists(p):
            if p not in self._image_cache:
                loaded_img = Image.open(p).convert("RGB")
                self._image_cache[p] = loaded_img if self.is_internvl else self._resize_image_optimally(loaded_img)
            return self._image_cache[p]
        return None

    def _resize_image_optimally(self, img: Image.Image, max_side: int = 1000):
        open_cv_image = np.array(img)
        height, width = open_cv_image.shape[:2]
        
        if max(width, height) > max_side:
            ratio = max_side / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            resized = cv2.resize(open_cv_image, new_size, interpolation=cv2.INTER_AREA)
            return Image.fromarray(resized)
        return img
    
    @staticmethod
    def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio

        return best_ratio

    def _dynamic_preprocess_logic(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        best_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        resized_img = image.resize((image_size * best_ratio[0], image_size * best_ratio[1]), Image.BICUBIC)
        processed_images = [resized_img.crop(( (i % best_ratio[0]) * image_size, (i // best_ratio[0]) * image_size,
                                               ((i % best_ratio[0]) + 1) * image_size, ((i // best_ratio[0]) + 1) * image_size ))
                            for i in range(best_ratio[0] * best_ratio[1])]
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size), Image.BICUBIC))
        return processed_images
    
    def _internvl_preprocess_image(self, item, input_size=448, max_num=6):
        pil_img = self._ensure_pil_image(item)
        if pil_img is None: return None, None
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        tiles = self._dynamic_preprocess_logic(pil_img, image_size=input_size, max_num=max_num)
        pixel_values = torch.stack([transform(t) for t in tiles])
        return pixel_values, pixel_values.size(0)
    
    def _hf_input_preprocessor(self, gen_args: GenerationArgs):
        engine_inputs = gen_args.engine_input if gen_args.is_batch_input else [gen_args.engine_input]
        processed_inputs = []

        for item in engine_inputs:
            messages = self._construct_chat(item, use_system_prompt=gen_args.use_system_prompt)
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=gen_args.add_generation_prompt
            )
            
            images = []
            if isinstance(item, list):
                for sub_item in item:
                    img = self._ensure_pil_image(sub_item)
                    if img: images.append(img)
            else:
                img = self._ensure_pil_image(item)
                if img: images.append(img)

            inputs = self.processor(
                text=[prompt],
                images=images if images else None,
                padding=True,
                return_tensors="pt"
            )
            processed_inputs.append(inputs)
        return processed_inputs
    
    def _vllm_openai_input_preprocessor(self, gen_args: GenerationArgs):
        sys_role = getattr(gen_args, "system_role", "system")
        def build(x):
            return self._construct_chat(
                x,
                use_system_prompt=gen_args.use_system_prompt,
                user_role=gen_args.user_role,
                model_role=gen_args.model_role,
                sys_role=sys_role,
            )
        items = gen_args.engine_input if gen_args.is_batch_input else [gen_args.engine_input]
        return [build(item) for item in items]

    def _vllm_inference(self, prompts, gen_args, gen_params) -> List[RequestOutput]:
        if gen_args.is_batch_input or len(prompts) == 1:
            model_output = self.model.chat(prompts, gen_params)
        elif not gen_args.is_batch_input and len(prompts) > 1:
            print(Fore.CYAN + f"[INFO] Sequential mode: Processing {len(prompts)} instances one by one...")
            model_output = []
            for prompt in tqdm(prompts, desc="Sequential Inference", leave=False):
                output = self.model.chat([prompt], gen_params, use_tqdm=False)
                model_output.extend(output)
        else:
            raise ValueError("Invalid input configuration for VLLM inference.")

        return model_output
    
    def _vllm_openai_inference(self, messages, params, response_format):
        attempts = 0
        while True:
            try:
                kwargs = {
                    "model": self.model_id,
                    "messages": messages,
                    **params
                }
                if response_format:
                    kwargs["response_format"] = response_format

                return self.model.chat.completions.create(**kwargs)

            except Exception as e:
                print(Fore.RED + f"[ERROR] vLLM-OpenAI Error: {e}")
                if attempts < 5:
                    time.sleep(2)
                    attempts += 1
                else:
                    return None

    def _openai_inference(self, prompt, gen_params, response_format):
        attempts = 0
        while True:
            try:
                kwargs = {
                    "model": self.model_id,
                    "input": prompt,
                    "max_output_tokens": gen_params.get("max_completion_tokens"),
                }
                effort = gen_params.get("reasoning", None)
                if effort:
                    kwargs["reasoning"] = {"effort": effort}

                if response_format is not None:
                    kwargs["response_format"] = response_format

                model_output = self.model.responses.create(**kwargs)
                break

            except OpenAIError as e:
                print(Fore.RED + f"[ERROR] OpenAIError: {e}")
                if "Please try again with a different prompt" in str(e):
                    model_output = None
                    break
                else:
                    if attempts < 50:
                        print(Fore.YELLOW + f"[INFO] Retrying in {(5 * attempts) % 60} seconds (attempt {attempts + 1})")
                        attempts += 1
                        time.sleep((5 * attempts) % 60)
                    else:
                        print(Fore.RED + f"[ERROR] Max attempts reached")
                        model_output = None
                        break
        return model_output
    
    def gemini_inference(self, contents, gen_params):
        attempts = 0
        self.gemini_rate_limiter.acquire()
        config = types.GenerateContentConfig(
                    temperature=gen_params.get("temperature"),
                    top_p=gen_params.get("top_p"),
                    candidate_count=gen_params.get("candidate_count"),
                    max_output_tokens=gen_params.get("max_output_tokens"),
                    stop_sequences=gen_params.get("stop_sequences"),
                )
        while True:
            try:
                
                response = self.model.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                )

                return response

            except Exception as e:
                print(Fore.RED + f"[ERROR] GeminiError: {e}")
                if attempts < 8:
                    t = min(2 ** attempts, 30)
                    print(Fore.YELLOW + f"[INFO] Retrying in {t} seconds (attempt {attempts+1})")
                    attempts += 1
                    time.sleep(t)
                else:
                    print(Fore.RED + "[ERROR] Max attempts reached.")
                    return None

    def gemini_parallel_inference(self, list_of_contents, gen_params):
        results = [None] * len(list_of_contents)

        max_workers = min(len(list_of_contents), 16)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.gemini_inference, contents, gen_params): idx
                for idx, contents in enumerate(list_of_contents)
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_idx),total=len(future_to_idx),desc="Gemini Inference"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(Fore.RED + f"[ERROR] Gemini parallel inference failed: {e}")
                    results[idx] = None

        return results

    def _openai_parallel_inference(self, prompts, gen_params, response_format):
        results = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(prompts), self.num_openai_workers)) as executor:
            futures = {executor.submit(self._openai_inference, prompt, gen_params, response_format): idx for idx, prompt in enumerate(prompts)}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(prompts), desc="OpenAI Inference"):
                idx = futures[future]
                result = future.result()
                idx = futures[future]
                results[idx] = result
            return results
    
    def _vllm_openai_parallel_inference(self, prompts, params, response_format):
        results = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(prompts), self.num_openai_workers)) as executor:
            futures = {executor.submit(self._vllm_openai_inference, p, params, response_format): i for i, p in enumerate(prompts)}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(prompts), desc="vLLM-OpenAI Inference"):
                results[futures[future]] = future.result()
        return results
    
    def gemini_parse_output(self, response, contents):
        if response is None:
            return [VLMInferenceOutput(output_seqs=[""], input_prompt=contents)]

        text = ""
        try:
            if hasattr(response, "text") and isinstance(response.text, str):
                text = response.text.strip()
        except Exception:
            pass

        return [VLMInferenceOutput(output_seqs=[text], input_prompt=contents)]

    def _vllm_parse_output(self, vllm_outputs):
        outputs = []
        for request_output in vllm_outputs:
            m = getattr(request_output, "metrics", None)
            latency = (m.finished_time - m.first_scheduled_time) if m else None
            prompt_logprobs = getattr(request_output, "prompt_logprobs", None)

            generation_output = sorted(request_output.outputs, key=lambda x: x.index)
            seqs, logprobs, cumulative, finish = [], [], [], []
            for s in generation_output:
                seqs.append(getattr(s, "text", ""))
                logprobs.append(getattr(s, "logprobs", None))
                cum = getattr(s, "cumulative_logprob", None)
                if cum is None: cum = getattr(s, "cumulative_logprobs", None)
                cumulative.append(cum)
                finish.append(getattr(s, "finish_reason", None))

            outputs.append(VLMInferenceOutput(
                output_seqs=seqs,
                output_objects=generation_output,
                input_prompt=getattr(request_output, "prompt", None),
                logprobs=logprobs,
                cumulative_output_logprobs=cumulative,
                finish_reasons=finish,
                latency=latency,
                prompt_logprobs=prompt_logprobs
            ))
        return outputs
    
    def _vllm_openai_parse_output(self, client_outputs, prompts):
        outputs = []
        for co, pr in zip(client_outputs, prompts):
            if co is None:
                outputs.append(VLMInferenceOutput(output_seqs=[""], input_prompt=pr))
                continue
            
            text = co.choices[0].message.content if co.choices else ""
            finish_reason = co.choices[0].finish_reason if co.choices else "unknown"
            usage = getattr(co, "usage", None)

            outputs.append(VLMInferenceOutput(
                output_seqs=[text],
                output_objects=co,
                usage=usage,
                input_prompt=pr,
                finish_reasons=[finish_reason]
            ))
        return outputs

    def _openai_parse_output(self, client_outputs, prompts):
        def text_from_response(resp):
            txt = getattr(resp, "output_text", None)
            if isinstance(txt, str) and txt.strip():
                return txt.strip()

            try:
                blocks = []
                for item in getattr(resp, "output", []) or []:
                    for block in getattr(item, "content", []) or []:
                        btxt = getattr(block, "text", None)
                        if isinstance(btxt, str):
                            blocks.append(btxt)
                if blocks:
                    return "".join(blocks).strip()
            except Exception:
                pass

            return ""

        def process_single_output(client_output, prompt):
            if client_output is None:
                return VLMInferenceOutput(output_seqs=[""], output_objects=None, input_prompt=prompt)

            usage = getattr(client_output, "usage", None)
            content_text = text_from_response(client_output)

            return VLMInferenceOutput(
                output_seqs=[content_text],
                output_objects=client_output,
                usage=usage,
                input_prompt=prompt,
                logprobs=None,
                finish_reasons=[getattr(client_output, "status", None)]  # "completed" / "incomplete"
            )

        if isinstance(client_outputs, list):
            return [process_single_output(co, pr) for co, pr in zip(client_outputs, prompts)]
        else:
            return [process_single_output(client_outputs, prompts)]


    def _construct_chat(self, msg_list, use_system_prompt=False, user_role="user", model_role="assistant", sys_role="system"):
        if isinstance(msg_list, list) and msg_list and isinstance(msg_list[0], dict) and "role" in msg_list[0] and "content" in msg_list[0]:
            return msg_list
        if isinstance(msg_list, str):
            if self.backend == "openai":
                return [{"role": user_role, "content": [{"type":"input_text","text": msg_list}]}]
            else:
                return [{"role": user_role, "content": [{"type":"text","text": msg_list}]}]

        parts = list(msg_list) if isinstance(msg_list, list) else [msg_list]
        chat = []
        if use_system_prompt and parts and isinstance(parts[0], str):
            if self.backend == "openai":
                chat.append({"role": sys_role, "content": [{"type":"input_text","text": parts.pop(0)}]})
            else:
                chat.append({"role": sys_role, "content": [{"type":"text","text": parts.pop(0)}]})

        if self.backend == "vllm":
            backend_key = "vllm"
        elif self.backend == "openai":
            backend_key = "openai-responses"
        else:
            backend_key = "chat-completions"

        chat.append({"role": user_role, "content": self._parts_to_content(parts, backend_key)})
        return chat
    
    def _construct_gemini_chat(self, msg_list):
        def process_item(item):
            if isinstance(item, Image.Image):
                return item
            if isinstance(item, str) and os.path.exists(item):
                if item not in self._image_cache:
                    loaded_img = Image.open(item).convert("RGB")
                    self._image_cache[item] = loaded_img 
                return self._image_cache[item]
            return str(item)

        if isinstance(msg_list, (str, Image.Image)):
            return [process_item(msg_list)]

        if isinstance(msg_list, list):
            return [process_item(i) for i in msg_list]
        
        return [str(msg_list)]

    
    def _parts_to_content(self, parts, backend: str):
        def pil_to_base64(img, key=None):
            if key and key in self._image_b64_cache:
                return self._image_b64_cache[key]
            
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode()
            
            if key:
                self._image_b64_cache[key] = b64
            return b64

        def resize_image_optimally(img: Image.Image, max_side: int = 1280):
            width, height = img.size
            if max(width, height) > max_side:
                ratio = max_side / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                return img.resize(new_size, Image.Resampling.BILINEAR)
            return img

        def ensure_pil(x):
            if isinstance(x, Image.Image):
                return resize_image_optimally(x)
            if isinstance(x, str) and os.path.exists(x):
                if x not in self._image_cache:
                    loaded_img = Image.open(x).convert("RGB")
                    self._image_cache[x] = resize_image_optimally(loaded_img)
                return self._image_cache[x]
            return None


        out = []
        has_image = False

        for p in parts:
            if isinstance(p, dict) and "type" in p:
                out.append(p)
                continue
            pil = ensure_pil(p)
            if pil is not None:
                has_image = True
                b64 = pil_to_base64(pil, key=p if isinstance(p, str) else None)
                data_url = f"data:image/jpeg;base64,{b64}"

                if backend == "vllm":
                    out.append({"type": "image_url", "image_url": {"url": data_url}})
                elif backend in ["chat-completions", "vllm-openai"]:
                    out.append({
                        "type": "image_url", 
                        "image_url": {"url": data_url}
                    })
                elif backend == "openai-responses":
                    b64 = pil_to_base64(pil, key=p if isinstance(p, str) else None)
                    data_url = f"data:image/jpeg;base64,{b64}"
                    out.append({"type": "input_image", "image_url": data_url})
                else:
                    b64 = pil_to_base64(pil)
                    data_url = f"data:image/jpeg;base64,{b64}"
                    out.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})
            else:
                text = p if isinstance(p, str) else str(p)

                if has_image and "internvl" in self.model_id.lower():
                    if "<image>" not in text:
                        text = "<image>\n" + text

                if backend == "openai-responses":
                    out.append({"type": "input_text", "text": text})
                elif backend in ["chat-completions", "vllm-openai", "vllm"]:
                    out.append({"type": "text", "text": text})
                else:
                    out.append({"type": "text", "text": text})
        return out


    
    def _shutdown(self):
        if self.backend == "vllm":
            destroy_model_parallel()
            destroy_distributed_environment()
            del self.model.llm_engine.model_executor
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

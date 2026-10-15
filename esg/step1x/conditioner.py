# Modified from Step1X-Edit (https://github.com/stepfun-ai/Step1X-Edit), licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
import torch
from qwen_vl_utils import process_vision_info
from torchvision.transforms import ToPILImage
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

VISION_END_ID = 151653  # <|vision_end|>
PREFIX_LEN = 217  # number of tokens of the system prefix below, which are dropped


class Qwen25VL_7b_Embedder(torch.nn.Module):
    def __init__(self, model_path, max_length=640, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.max_length = max_length
        self.dtype = dtype
        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=324 * 28 * 28)

        self.prefix = """Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n
Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n
Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt:"""

    @torch.no_grad()
    def forward(self, captions, ref_images):
        """Returns embeddings [b, max_length, 3584], masks [b, max_length] and the token strings of each caption."""
        embs = torch.zeros(
            len(captions), self.max_length, self.model.config.hidden_size, dtype=self.dtype, device=self.device
        )
        masks = torch.zeros(len(captions), self.max_length, dtype=torch.long, device=self.device)
        tokens_list = []

        for idx, (txt, imgs) in enumerate(zip(captions, ref_images)):
            messages = [{"role": "user", "content": []}]
            messages[0]["content"].append({"type": "text", "text": f"{self.prefix}"})
            messages[0]["content"].append({"type": "image", "image": ToPILImage()(imgs)})
            messages[0]["content"].append({"type": "text", "text": f"{txt}"})

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
            old_input_ids = inputs.input_ids

            # The image is encoded with its pad tokens, but the token sequence after <|vision_end|> comes from the text
            # processed without the image (same as the official Step1X-Edit implementation).
            new_input_ids = self.processor(
                text=[text], images=None, videos=None, padding=True, return_tensors="pt"
            ).input_ids
            idx1 = (old_input_ids == VISION_END_ID).nonzero(as_tuple=True)[1][0]
            idx2 = (new_input_ids == VISION_END_ID).nonzero(as_tuple=True)[1][0]
            input_ids = (
                torch.cat([old_input_ids[0, :idx1], new_input_ids[0, idx2:]], dim=0).unsqueeze(0).to(self.device)
            )
            attention_mask = (input_ids > 0).long()

            tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids[0])
            tokens_list.append(tokens[PREFIX_LEN:][: self.max_length])

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=inputs.pixel_values.to(self.device),
                image_grid_thw=inputs.image_grid_thw.to(self.device),
                output_hidden_states=True,
            )
            emb = outputs["hidden_states"][-1]

            n = min(self.max_length, emb.shape[1] - PREFIX_LEN)
            embs[idx, :n] = emb[0, PREFIX_LEN:][: self.max_length]
            masks[idx, :n] = 1

        return embs, masks, tokens_list

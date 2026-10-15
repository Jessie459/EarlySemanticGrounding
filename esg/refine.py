"""Editing-instruction refinement with the prompt rewriter ("thinker") of Step1X-Edit v1.2 (ReasonEdit).

Re-implements `Step1XEditThinker.think` from the Step1X-Edit v1.2 diffusers pipeline with plain transformers.
The prompts are taken from Step1X-Edit (https://github.com/stepfun-ai/Step1X-Edit), licensed under the
Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
"""

import re

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SYSTEM_PROMPT = "You are an Edit Task Instruction Rewriter. Your role is to maintain the original meaning while transforming vague, colloquial, or imprecise instructions into clear, standardized, and actionable ones. If the instructions are already sufficiently clear, they should be returned without modification."

USER_PROMPT = """You are an 'editing task instruction rewriter'. Your task is:
While keeping the meaning completely unchanged, convert the user's vague, colloquial, or informal editing instruction into a clear, standard, and actionable instruction.
When the user's input contains complex meanings, break it down into two to three simple, clear editing instructions. Ensure these instructions do not conflict with each other and retain only the meaningful ones. Output only the rewritten content.
The final editing instruction should preferably contain only one editing action. If the meaning is too complex to be described with a single action, the total number of actions must not exceed three.
If the input includes explicit text modifications, enclose the text to be changed in double quotes.
Editing types mainly include: object addition, object replacement, object deletion, color modification, background replacement, style change, material change, and text replacement.
Where possible, the final instruction should include the editing location, the action (see the editing types), and the object being edited. Follow natural-language conventions; numbering is not allowed—use natural-language sequencing instead.
If the user's input requires understanding and reasoning, think about where the final image will change based on the user's description, rather than returning a result prior to such reasoning.
Rewrite in the style of the following examples.
Example 1: User input: 'Symptoms of potassium deficiency in leaves' → Rewritten: 'Leaves turn yellow, and the tips dry out.'
Example 2: User input: 'Make the bread look over-fried' → Rewritten: 'Change the bread’s color to a dark, toasted brown.'
Example 3: User input: 'Photoshop me a boyfriend; make it a two-person photo' → Rewritten: 'Add a man next to the woman.'
Example 4: User input: 'Improve this image, remove the distant power lines, keep a realistic style' → Rewritten: 'Remove the power lines.'
Example 5: User input: 'Change “TOFEL” in the image to “IELTS”' → Rewritten: 'Replace the text "TOFEL" with "IELTS".'
Example 6: User input: 'Make it an ID photo.' → Rewritten: 'Replace the background with white.'
Example 7: User input: 'Change the tie to black.' → Rewritten: 'Change the tie color to black.'
Example 8: User input: 'Remove the watermark in the bottom-right corner.' → Rewritten: 'Remove the bottom-right watermark.'
Example 9: User input: 'Increase hair volume. Change the hairstyle to long hair. Make the person look gentler.' → Rewritten: 'Increase the amount of hair to make the person look gentler.'
User input instruction:{instruction}
Your returned instruction must be in {language}. Your returned instruction must be in {language}. Your returned instruction must be in {language}.
The returned instruction is:"""


class InstructionRefiner:
    def __init__(self, model_root, device="cuda"):
        """model_root: the stepfun-ai/Step1X-Edit-v1p2 checkpoint (only `text_encoder/` and `processor/` are used)."""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            f"{model_root}/text_encoder", torch_dtype=torch.bfloat16
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(f"{model_root}/processor", min_pixels=262144, max_pixels=802816)

    @torch.no_grad()
    def __call__(self, image: Image.Image, instruction: str) -> str:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        language = "中文" if len(re.findall(r"[一-鿿]", instruction)) > 1 else "英文"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_PROMPT.format(instruction=instruction, language=language)},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output[0].strip()

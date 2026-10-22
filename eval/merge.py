# Merge a LoRA adapter for standalone evaluation; preserve adapter tokenizer and chat template.
import os, shutil, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

p = argparse.ArgumentParser()
p.add_argument("--base_model", type=str, default=None)
p.add_argument("--adapter_path", type=str, default=None)
p.add_argument("--save_path", type=str, default=None)
args_cli = p.parse_args()


base_model = args_cli.base_model
adapter_path = args_cli.adapter_path
save_path = args_cli.save_path

# load base in full precision (not 4/8-bit) so merge works
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)   # attaches LoRA

# sanity check: should show a LoRA config, not empty
print(model.peft_config)

# merge and save
model = model.merge_and_unload()
model.save_pretrained(save_path)

# save tokenizer/template from the ADAPTER (falls back to base if missing)
tok_src = adapter_path if os.path.exists(os.path.join(adapter_path, "tokenizer.json")) else base_model
tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
tokenizer.save_pretrained(save_path)
ct = os.path.join(adapter_path, "chat_template.jinja")
if os.path.exists(ct): shutil.copy(ct, save_path)

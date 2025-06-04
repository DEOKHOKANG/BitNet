import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train QLoRA model")
    parser.add_argument('--model', type=str, default='microsoft/BitNet-b1.58-2B-4T')
    parser.add_argument('--dataset', type=str, default='wikitext', help='HF dataset name')
    parser.add_argument('--output', type=str, default='qlora-bitnet')
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', load_in_4bit=True)
    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    model = get_peft_model(base_model, lora_config)

    dataset = load_dataset(args.dataset, 'wikitext-2-raw-v1', split='train[:1%]')

    def tokenize(sample):
        return tokenizer(sample['text'], truncation=True, padding='max_length', max_length=128)

    tokenized = dataset.map(tokenize, batched=True)

    model.train()
    for batch in tokenized.shuffle(seed=42).with_format('torch').select(range(100)):
        outputs = model(**{k: batch[k] for k in ['input_ids', 'attention_mask']}, labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == '__main__':
    main()

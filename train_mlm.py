# Run with: accelerate launch --config_file accelerate.cfg train_mlm.py

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, get_constant_schedule_with_warmup
from datasets import load_dataset

# Minimal params
model_name = 'xlm-roberta-base'
batch_size = 16
num_warmup_steps = 0
max_length = 100
# gradient_accumulation_steps = 1
gradient_accumulation_steps = 256
use_orig_params = True
# use_orig_params = False

# Load tokenizer and collate function
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

# Collate function for Masked Language Model
def collate_fn(items):
    texts = [i['text'] for i in items]
    tokenized = tokenizer(texts, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True, return_special_tokens_mask=True)

    input_ids, labels = data_collator.torch_mask_tokens(
        inputs=tokenized["input_ids"],
        special_tokens_mask=tokenized["special_tokens_mask"],
    )
    return {"input_ids": input_ids, "labels": labels, "attention_mask": tokenized["attention_mask"]}

# Get model
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Get initialised accelerator and re-initialise with FSDP plugin using use_orig_params
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
accelerator.state.fsdp_plugin.use_orig_params = use_orig_params
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, fsdp_plugin=accelerator.state.fsdp_plugin)

# Wrap model with FSDP
model = accelerator.prepare(model)

# Load dataset and make data_loader
dataset = load_dataset('cardiffnlp/tweet_sentiment_multilingual', 'all', split='train')
data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# Get optimiser
optimizer = AdamW(model.parameters())

# Get scheduler
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

# Wrap data_loader, optimizer and scheduler with FSDP
data_loader, optimizer, scheduler = accelerator.prepare(data_loader, optimizer, scheduler)

# Training loop
for batch in data_loader:
    # print(batch)
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        print(loss)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import pandas as pd

# Load data
df = pd.read_csv("Vishay_Trained_Data.txt", sep='\t', encoding='iso-8859-1')
input_list = df['MPN'].to_list()
output_list = df['SE_PART'].to_list()

# Load pre-trained model and tokenizer
model_name = 't5-base'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Prepare data
def preprocess_data(original, corrected):
    inputs = ["correct: " + sentence for sentence in original]
    targets = corrected
    input_encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    target_encodings = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
    return input_encodings, target_encodings

input_encodings, target_encodings = preprocess_data(input_list, output_list)

# Fine-tune the model
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
batch_size = 400
epochs = 14
accumulation_steps = 16  # Adjust this to accumulate gradients over multiple steps

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Train the model
model.train()
for epoch in range(epochs):
    for i in range(0, len(input_encodings.input_ids), batch_size):
        input_ids = input_encodings.input_ids[i:i + batch_size]
        labels = target_encodings.input_ids[i:i + batch_size]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i // batch_size + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch {epoch + 1}/{epochs}, Step {i // batch_size + 1}/{len(input_encodings.input_ids) // batch_size}, Loss: {loss.item() * accumulation_steps}",flush=True)

# Save the fine-tuned model
model.save_pretrained("./corrector_model")
tokenizer.save_pretrained("./corrector_model")

# Load the model for inference
model = T5ForConditionalGeneration.from_pretrained("./corrector_model")
tokenizer = T5Tokenizer.from_pretrained("./corrector_model")

# Example inference
def correct_text(input_text):
    input_ids = tokenizer("correct: " + input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Example usage
print(correct_text("CRCW0603180RFK(EA, EB or EC)"))
print(correct_text("ZY 27 DO-41"))
print(correct_text("ZM4746A-GS08-CUTTAPE"))

#Run below command in below terminal
#nohup python3 Vishay_Train_Model.py &
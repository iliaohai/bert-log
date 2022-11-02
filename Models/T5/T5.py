from transformers import T5Tokenizer, T5EncoderModel

# t5文本表示
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5EncoderModel.from_pretrained("t5-base")
input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
outputs = model(input_ids=input_ids)
last_hidden_states = outputs.last_hidden_state


print(outputs)
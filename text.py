from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode text
inputs = tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)
text_embeddings = model(**inputs).last_hidden_state[:, 0, :]

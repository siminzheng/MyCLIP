from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_texts(text_list, max_len=32):
    tokens = tokenizer(text_list, padding='max_length', max_length=max_len, truncation=True, return_tensors="pt")
    return tokens.input_ids

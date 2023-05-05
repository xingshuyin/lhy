from transformers import BertTokenizer

sents = ["选择珠江花园的原因就是方便", "有电动扶梯直接到达海边"]
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-chinese')
tokenizer.encode_plus(text=sents[0], text_pair=sents[1], max_length=30, truncation=True, padding=True, add_special_tokens=True, return_token_type_ids=True, return_attention_mask=True, return_special_tokens_mask=True, return_length=True)

from transformers import AutoTokenizer, BertForNextSentencePrediction
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

# prompt = "David is referring to David Beckham, the husband of Victoria Beckham and the father of Brooklyn, Romeo, Cruz, and Harper Seven."
# next_sentence = "David Robert Joseph Beckham , (born 2 May 1975) is an English former footballer. He has played for Manchester United, Preston North End, Real Madrid, Milan, Los Angeles Galaxy, Paris Sa."
prompt = "David Robert Joseph Beckham OBE (UK: /ˈbɛkəm/;[6] born 2 May 1975) is an English former professional footballer, the current president and co-owner of Inter Miami CF and co-owner of Salford City.[7] Known for his range of passing, crossing ability and bending free-kicks as a right winger,"
next_sentence = "Beckham has frequently been hailed as one of the greatest and most recognisable midfielders of his generation, as well as one of the best set-piece specialists of all time."

encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

outputs = model(**encoding, labels=torch.LongTensor([1]))
logits = outputs.logits
assert logits[0, 0] < logits[0, 1]  # next sentence was random
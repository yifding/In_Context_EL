from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

if __name__ == '__main__':
    max_length = 256

    premise = "Liu Chao-shiuan (Chinese: 劉兆玄; pinyin: Liú Zhàoxuán; born 10 May 1943) is a Taiwanese educator and politician. He is a former president of the National Tsing Hua University (1987–1993) and Soochow University (2004–2008) and a former Premier of the Republic of China (2008–2009)."
    hypothesis = "Liu Chao-shiuan is/was an agent"

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    # Note:
    # "id2label": {
    #     "0": "entailment",
    #     "1": "neutral",
    #     "2": "contradiction"
    # },

    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    print("Premise:", premise)
    print("Hypothesis:", hypothesis)
    print("Entailment:", predicted_probability[0])
    print("Neutral:", predicted_probability[1])
    print("Contradiction:", predicted_probability[2])
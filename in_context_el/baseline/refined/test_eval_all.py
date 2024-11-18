from refined.inference.processor import Refined
from refined.evaluation.evaluation import eval_all


# refined = Refined.from_pretrained(model_name='wikipedia_model',
#                                   entity_set='wikipedia',
#                                   use_precomputed_descriptions=True)

# print('ED results (with model not fine-tuned on AIDA)')
# eval_all(refined=refined, el=False)


'''
Dataset name: AIDA

****************
************
f1: 0.8743
accuracy: 0.8481
gold_recall: 0.9785
p: 0.9021
r: 0.8481
num_gold_spans: 4464
************

*****************************


Evaluating on MSNBC: 20it [00:01, 15.36it/s]
*****************************


Dataset name: MSNBC

****************
************
f1: 0.9450
accuracy: 0.9370
gold_recall: 0.9954
p: 0.9531
r: 0.9370
num_gold_spans: 651
************

*****************************


Evaluating on AQUAINT: 50it [00:02, 24.92it/s]
*****************************


Dataset name: AQUAINT

****************
************
f1: 0.9191
accuracy: 0.8892
gold_recall: 0.9515
p: 0.9511
r: 0.8892
num_gold_spans: 722
************

*****************************


Evaluating on ACE2004: 36it [00:01, 34.76it/s]
*****************************


Dataset name: ACE2004

****************
************
f1: 0.9139
accuracy: 0.8814
gold_recall: 0.9091
p: 0.9489
r: 0.8814
num_gold_spans: 253
************

*****************************


Evaluating on CWEB: 320it [00:24, 13.06it/s]
*****************************


Dataset name: CWEB

****************
************
f1: 0.7819
accuracy: 0.7396
gold_recall: 0.9587
p: 0.8293
r: 0.7396
num_gold_spans: 11038
************

*****************************


Evaluating on WIKI: 320it [00:10, 30.44it/s]
*****************************


Dataset name: WIKI

****************
************
f1: 0.8876
accuracy: 0.8600
gold_recall: 0.9373
p: 0.9169
r: 0.8600
num_gold_spans: 6773
************

*****************************
'''


refined = Refined.from_pretrained(model_name='aida_model',
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=True)

print('ED results (with model not fine-tuned on AIDA)')
eval_all(refined=refined, el=False)
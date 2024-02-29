# In_Context_EL
This is our coding implementation for the paper:

```bibtex
@inproceedings{ding2024chatel,
 title={ChatEL: ChatEL: Entity Linking with Chatbots},
 author={Yifan Ding, Qingkai Zeng, Tim Weninger},
 booktitle={COLING-LREC},
 year={2024}
}

@article{ding2024entgpt,
  title={EntGPT: Linking Generative Large Language Models with Knowledge Bases},
  author={Yifan Ding, Amrit Poudel, Qingkai Zeng, Tim Weninger, Balaji Veeramani, Sanmitra Bhattacharya},
  journal={arXiv preprint arXiv:2402.06738},
  year={2024}
}
```

Author: Yifan Ding (yding4@nd.edu)

## installation
1. this repo is built on top of [REL](https://github.com/yifding/REL) and [BLINK](https://github.com/facebookresearch/BLINK). Install REL and BLINK from the corresponding websites.
2. install In_context_EL as a package
```bash
git clone https://github.com/yifding/In_Context_EL.git
cd In_Context_EL
pip install -e ./
```
3. prepare opeai API key, create a file named ```openai_key.py``` with openai API under the directory of In_Context_EL/in_context_el
```python
OPENAI_API_KEY = "sk-*****"
```
4. download dataset files from [google drive](https://drive.google.com/file/d/1XyOSV90G7sLxd9PqA_MvUbzV_tqyLOYn/view?usp=sharing), unzip the file under the directory of In_Context_EL/data

## run the code of ChatEL & EntGPT-P (GPT3.5 results)
0. donwload the running script with running outputs from [google drive](https://drive.google.com/file/d/1cs5jGoVJcV32XuJkxrx-OZv6GDOvFO32/view?usp=sharing)
1. obtain REL entity candidates
```
sh In_Context_EL/RUN_FILES/public/rel_blink_candidates/KORE50.sh
```

2. obtain BLINK and REL entity candidates
```
sh In_Context_EL/RUN_FILES/public/rel_blink_candidates/KORE50.sh
```

3. obtain summary and context augmentation prompt response
```
sh In_Context_EL/RUN_FILES/public/mention_prompt/KORE50.sh
```

4. obtain multi-choice selection prompt response
```
sh In_Context_EL/RUN_FILES/public/entity_candidate_prompt/KORE50.sh
```

5. evaluation
```
sh In_Context_EL/RUN_FILES/public/evaluation/KORE50.sh
```

## Reproduced Performances (GPT-3.5, zero-shot)
| Dataset | Precision | Recall | F1 | 
| ------------- | ------------- | ------------- | ------------- | 
| AIDA-YAGO | 0.830 | 0.812 | 0.821 |
| msnbc | 0.900 | 0.835 | 0.866 |
| aquaint | 0.830 | 0.757 | 0.791 |
| ACE 2004 | 0.913 | 0.856 | 0.884 |
| clueweb - WNED-CWEB (CWEB) | 0.767 | 0.662 | 0.711 |
| wikipedia - WNED-WIKI (WIKI) | 0.800 | 0.745 | 0.771 |
|KORE50| 0.814 | 0.639 | 0.716 |
|OKE15| 0.847 | 0.701 | 0.767 |
|OKE16| 0.855 | 0.701 | 0.770 |
|Reuters-128| 0.840 | 0.737 | 0.785 |
|RSS-500| 0.850 | 0.769 | 0.808 |





## Accelerating KBQA via Logical-Question Bidirectional Reranking

## Overview

![rankkbqa-over](.\figs\rankkbqa-over.png)

## Preliminary Setup

All of the datasets use Freebase as the knowledge source. Please follow [Freebase Setup](https://github.com/dki-lab/Freebase-Setup) to set up a Virtuoso triplestore service. Specific operations can be found in this repository。



## Environment Setup

**Fine-tuning  Environment**

```
pip install -r requirement_finetuning.txt
```

**Inference Environment**

```
pip install -r requirement_inference.txt
```

**FACC1 mentions for Entity Retrieval**

Download the mention information (including processed FACC1 mentions with [link](https://drive.google.com/file/d/1NZFaeytHLqceoHVL09gWSGNPYWug3ebP/view?usp=sharing) and all entity alias with [link](https://drive.google.com/file/d/1AwuHSl9WmiEHDlCRpkiVFa0mxz62deVu/view?usp=sharing) in Freebase) to `data/common_data/facc1/`.

## Dataset

Experiments are conducted on two  standard KBQA dataset WebQSP, CWQ. In order to facilitate the subsequent experiments, we provide the version with some simple preprocessing in addition to the original dataset.

```
RankKBQA/
└── data/
    ├── WebQSP                  
        ├── generation
        ├── origin
        └── sexpr
    ├── CWQ                  
        ├── generation
        ├── origin
        └── sexpr
```

## Models 

| Models              | Source                                                       |
| ------------------- | ------------------------------------------------------------ |
| Llama-3-8B-Instruct | [URL](https://huggingface.co/meta-llama/Meta-Llama-3-8B)     |
| BGE-Reranker-v2-m3  | [URL](https://huggingface.co/BAAI/bge-reranker-v2-m3)        |
| SimCSE              | [URL](https://huggingface.co/princeton-nlp/unsup-simcse-roberta-large) |
| T5-Family           | [URL](https://huggingface.co/google-t5)                      |



## Experiment Setup

Since the operations for both datasets are basically the same, we take WebQSP as an example here, and for CWQ, the same steps can be performed under the CWQ directory.

**Preliminary Logical Form Generation**

```
nohup python -u WebQSP/Llama_Origin_Sexpr_Generation.py >> log/WebQSP_pred_Sexpr_Llama3_Sexpr.txt 2>&1 &
```

The generated results will be stored in `results/gen/WebQSP_test/evaluation_beam_llama/generated_predictions.jsonl`

**Format preliminary generation files**

```
python run_generator_final.py --data_file_name results/gen/WebQSP_test/evaluation_beam_llama/generated_predictions.jsonl
```

The formatted result will be saved in

`results/gen/WebQSP_test/evaluation_beam_llama/beam_test_top_k_predictions.json`

**Training a transcription model**

Since the WebQSP dataset does not provide a validation set, we randomly sample 200 entries from the training set as the validation set.

```
sh WebQSP/run_reverse.sh train query
```

models will be saved in `exps/reverse_WebQSP_query/output`

**Prediction and Bidirectional Rerank on the test set**

```
sh WebQSP/run_reverse.sh predict exps/reverse_WebQSP_query/output test
```

The final result will be saved in

`results/reverse/WebQSP_test/evaluation_beam_llama/top_k_predictions.json`

**Evaluation**

```
CUDA_VISIBLE_DEVICES=0 nohup python -u eval_final_webqsp.py --dataset WebQSP --pred_file results/reverse/WebQSP_test/evaluation_beam_llama/top_k_predictions.json >> log/WebQSP_pred_final_llama_sexpr.txt 2>&1 &
```


"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import sys

folder = os.path.abspath(os.path.dirname(__file__))
root = os.path.dirname(folder)
sys.path.append(root)

import json
import logging
import os
import sys
import timeit
from os.path import join
from dataclasses import dataclass, field
from functools import partial
# import spacy
from rank_bm25 import BM25Okapi
from operator import itemgetter
from typing import Callable, Dict, List, Optional, Tuple, Iterable
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
# from simcse import SimCSE

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartTokenizer,
    HfArgumentParser,
    T5Tokenizer,
    TrainingArguments,
    set_seed,
    EvalPrediction,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)

from components.gen_dataset_manager import (
    load_and_cache_gen_examples,
    load_and_cache_reverse_examples,
    ListDataset,
    load_and_cache_reverse_examples_eval
)
from components.gen_dataset import generation_collate_fn, GenerationFeature
from components.generation_trainer import GenerationTrainer

logger = logging.getLogger(__name__)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '123456'
os.environ["WANDB_HTTP_TIMEOUT"] = "300"
os.environ["WANDB_INIT_TIMEOUT"] = "600"
os.environ["WANDB_DEBUG"] = "true"


def dump_json(obj, fname, indent=None):
    with open(fname, 'w') as f:
        return json.dump(obj, f, indent=indent)


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Parameters:
        label_smoothing (:obj:`float`, `optional`, defaults to 0):
            The label smoothing epsilon to apply (if not zero).
        sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to SortishSamler or not. It sorts the inputs according to lenghts in-order to minimizing the padding size.
        predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """
    eval_strategy: Optional[str] = field(default='no', metadata={
        "help": " The evaluation strategy to adopt during training(no、steps、epoch)"})
    warmup_ratio: Optional[float] = field(default=0.0, metadata={"help": "The warmup ratio"})
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_type: str = field(
        metadata={"help": "type of the model, t5 or bart"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset: str = field(default=None, metadata={"help": "dataset id"})
    train_file: str = field(default=None, metadata={"help": "path to training file"})
    predict_file: str = field(default=None, metadata={"help": "path to predict file"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='hfcache', metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    # freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    # freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})
    max_source_length: Optional[int] = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    # top_k_candidates: Optional[int] = field(default=5, metadata={"help": "# top k candidates used for generation."})
    do_lower_case: bool = field(default=False)
    overwrite_cache: bool = field(default=False)

    # local_rank: Optional[int] = field(default=-1,metadata={"help": "local_rank for distributed training on gpus."})


def _pad_tensors_to_max_len(tensor, max_length, pad_token_id):
    padded_tensor = pad_token_id * torch.ones(
        (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, : tensor.shape[-1]] = tensor
    return padded_tensor


def calculate_borda_count(pred_framework, scored_data):
    borda_scores = defaultdict(int)


    for rank, pred_fw in enumerate(pred_framework):
        borda_scores[pred_fw] += (len(pred_framework) - rank) * 0.6


    scored_data_sorted = sorted(scored_data, key=itemgetter(2), reverse=True)
    for rank, (_, pred_fw, _) in enumerate(scored_data_sorted):
        borda_scores[pred_fw] += (len(scored_data_sorted) - rank) * 0.4


    final_sorted_data = sorted(
        scored_data, key=lambda x: borda_scores[x[1]], reverse=True
    )

    return final_sorted_data


def BM25_ranker(query, Ts_que):
    nlp = spacy.load("en_core_web_sm")

    questions = [item[0] for item in Ts_que]

    tokenized_corpus = []
    for question in questions:
        nlp_question = nlp(question)
        tokenized_corpus.append([token.lemma_ for token in nlp_question if not token.is_stop and not token.is_punct and token.is_alpha])
        # tokenized_corpus.append([token.lemma_ for token in nlp_question])

    # bm25 = BM25Okapi(tokenized_corpus)
    bm25 = BM25Okapi(tokenized_corpus, k1=1.2, b=0.6)

    tokenized_query = nlp(query)
    # tokenized_query = [token.lemma_ for token in tokenized_query]
    tokenized_query = [token.lemma_ for token in tokenized_query if not token.is_stop and not token.is_punct and token.is_alpha]

    scores = bm25.get_scores(tokenized_query)


    question_logic_score_pairs = list(zip(questions, [item[1] for item in Ts_que], scores))

    # sorted_pairs = sorted(question_logic_score_pairs, key=lambda x: x[2], reverse=True)
    return question_logic_score_pairs


def simcse_ranker(query, Ts_que):
    model = SimCSE("unsup-simcse-roberta-large")

    questions = [q[0] for q in Ts_que]
    logical_forms = [q[1] for q in Ts_que]

    similarities = model.similarity([query.lower()], [q.lower() for q in questions])

    scores = similarities[0]


    question_logic_score_pairs = list(zip(questions, logical_forms, scores))

    return question_logic_score_pairs


def run_prediction(train_args, model_args, dataset, model, tokenizer, output_prediction=False):
    if not os.path.exists(train_args.output_dir) and train_args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    max_length = (
        model.config.max_generate_length
        if hasattr(model.config, "max_generate_length")
        else model.config.max_position_embeddings
    )
    num_beams = model.config.num_beams
    pad_token_id = model.config.pad_token_id
    # multi-gpu evaluate
    # only allow using one gpu here
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", train_args.per_device_eval_batch_size)

    start_time = timeit.default_timer()
    model = model.to(train_args.device)
    model.eval()

    ranker_tokenizer = AutoTokenizer.from_pretrained('bge-reranker-v2-m3')
    ranker_model = AutoModelForSequenceClassification.from_pretrained('bge-reranker-v2-m3')
    ranker_model.eval()

    output_list = []
    ex_cnt = 0
    contains_ex_cnt = 0
    real_total = 0
    add_prefix_space = isinstance(tokenizer, BartTokenizer)

    for ex in tqdm(dataset, desc='Evaluating', total=len(dataset)):
        qid = ex.qid
        query = ex.query
        label = ex.gt.normed_expr
        candidates = ex.candidates
        dst_text = query
        pred_outputs = OrderedDict()
        Ts_que = []
        print("\n=========================")
        print("QID:{}".format(qid))
        print("Question:{}".format(query))
        print("Label:{}".format(label))
        for candidate in tqdm(candidates, total=len(candidates), desc='Decoding'):
            src_text = candidate
            feature = []
            all_predictions = []
            all_labels = []
            if add_prefix_space:
                batch_encoding = tokenizer.prepare_seq2seq_batch(
                    [src_text],
                    [dst_text],
                    max_length=model_args.max_source_length,
                    max_target_length=model_args.max_target_length,
                    return_tensors="pt",
                    add_prefix_space=add_prefix_space,
                ).data
            else:
                batch_encoding = tokenizer.prepare_seq2seq_batch(
                    [src_text],
                    [dst_text],
                    max_length=model_args.max_source_length,
                    max_target_length=model_args.max_target_length,
                    return_tensors="pt",
                ).data
            input_ids, labels = batch_encoding['input_ids'][0], batch_encoding['labels'][0]
            feature.append(GenerationFeature(ex, input_ids, labels))
            batch = generation_collate_fn(feature, tokenizer)
            batch = {k: v.to(train_args.device) for k, v in batch.items()}
            labels = batch.pop('labels')
            [all_labels.append(l.cpu().numpy()) for l in labels]

            with torch.no_grad():
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,
                    num_beams=num_beams,
                    num_return_sequences=1,
                    max_length=max_length,
                )

                [all_predictions.append(p.cpu().numpy()) for p in generated_tokens]

            for pred in all_predictions:
                decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
                print(decoded_pred)
                pred_outputs[candidate] = decoded_pred
                Ts_que.append((decoded_pred, candidate))

        pairs = [[query, que[0]] for que in Ts_que]


        # bge-reranker-v2-m3
        with torch.no_grad():
            inputs = ranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = ranker_model(**inputs, return_dict=True).logits.view(-1, ).float()

        scored_data = [(Ts_que[i][0], Ts_que[i][1], scores[i].item()) for i in range(len(scores))]

        # scored_data = BM25_ranker(query,Ts_que)

        #scored_data = simcse_ranker(query,Ts_que)


        # sorted_data = sorted(scored_data, key=lambda x: x[2], reverse=True)
        # print("Final Sorted Data (Borda Count):{}".format(sorted_data))


        final_sorted_data = calculate_borda_count(candidates, scored_data)
        print("Final Sorted Data (Borda Count):{}".format(final_sorted_data))

        sorted_expressions = [item[1] for item in final_sorted_data]
        print("Pred:{}".format(sorted_expressions))

        if sorted_expressions[0].lower() == label.lower():
            ex_cnt += 1
        if any([x.lower() == label.lower() for x in sorted_expressions]):
            contains_ex_cnt += 1

        if label.lower() != 'null':
            real_total += 1

        output_list.append({
            'predictions': sorted_expressions,
            'gen_label': label,
        })

        print(f"""ex_cnt:{ex_cnt}, 
                  real_ex_rate:{ex_cnt / real_total}, 
                  contains_ex_cnt:{contains_ex_cnt}, 
                  real_contains_ex_rate:{contains_ex_cnt / real_total}
                  """)

    if output_prediction:
        dump_json(output_list, join(train_args.output_dir, 'top_k_predictions.json'), indent=4)
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    return {'total': real_total, 'ex': ex_cnt / real_total}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    # if training_args.local_rank != -1:
    #     dist.init_process_group(backend='nccl', init_method='env://', rank=training_args.local_rank, world_size=1)
    # model_args.local_rank = -1
    # training_args.local_rank = -1
    model_args.local_rank = training_args.local_rank
    # model_args.output_dir = training_args.output_dir
    # model_args.n_gpu = training_args.n_gpu
    # model_args.eval_batch_size = training_args.per_device_eval_batch_size * max(1, training_args.n_gpu)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    '''

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    '''
    # use task specific params
    # use_task_specific_params(model, data_args.task)

    # set num_beams for evaluation
    if model_args.eval_beams is not None:
        model.config.num_beams = model_args.eval_beams
    assert model.config.num_beams >= 1, f"got eval_beams={model.config.num_beams}. Need an integer >= 1"
    model_args.logger = logger

    # set max length for generation
    model.config.max_generate_length = model_args.max_target_length

    def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
        def non_pad_len(tokens: np.ndarray) -> int:
            return np.count_nonzero(tokens != tokenizer.pad_token_id)

        def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
            pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
            pred_str = lmap(str.strip, pred_str)
            label_str = lmap(str.strip, label_str)
            return pred_str, label_str

        # with decoding
        def _exact_match_metrics(pred: EvalPrediction) -> Dict:
            # print(pred)
            pred_str, label_str = decode_pred(pred)
            ex = sum([a == b for (a, b) in zip(pred_str, label_str)]) / len(pred_str)
            result = {'ex': ex}
            gen_len = np.mean(lmap(non_pad_len, pred.predictions))
            result.update({"gen_len": gen_len})
            return result

        # without decoding
        def exact_match_metrics(pred: EvalPrediction) -> Dict:
            # print(pred)
            # pred_str, label_str = decode_pred(pred)
            ex = np.sum(np.all(pred.label_ids == pred.predictions, axis=1)) / pred.label_ids.shape[0]
            # for a, b in zip(pred.label_ids, pred.predictions):
            #     print(a)
            #     print(b)
            # exit()
            result = {'ex': ex, 'num_total': pred.label_ids.shape[0]}
            gen_len = np.mean(lmap(non_pad_len, pred.predictions))
            result.update({"gen_len": gen_len})
            return result

        # with decoding
        def bleu_rouge_metrics(pred: EvalPrediction) -> Dict:
            gen_len = np.mean(lmap(non_pad_len, pred.predictions))
            pred_str, label_str = decode_pred(pred)
            
            # Calculate BLEU score
            bleu_scores = []
            for pred, label in zip(pred_str, label_str):
                bleu_scores.append(sentence_bleu([label.split()], pred.split()))
            avg_bleu = np.mean(bleu_scores)

            # Calculate ROUGE score
            rouge = Rouge()
            rouge_scores = rouge.get_scores(pred_str, label_str, avg=True)

            return {
                'bleu': avg_bleu,
                'rouge-1': rouge_scores['rouge-1']['f'],
                'rouge-2': rouge_scores['rouge-2']['f'],
                'rouge-l': rouge_scores['rouge-l']['f'],
                'gen_len': gen_len,
            }

        # compute_metrics_fn = summarization_metrics if "summarization" in task_name else translation_metrics
        # compute_metrics_fn = exact_match_metrics
        compute_metrics_fn = bleu_rouge_metrics
        return compute_metrics_fn

    # Get datasets
    if training_args.do_train:
        train_dataset = ListDataset(load_and_cache_reverse_examples(model_args, tokenizer, evaluate=False))
    else:
        train_dataset = ListDataset([])
    if training_args.do_eval:
        eval_dataset = ListDataset(load_and_cache_reverse_examples(model_args, tokenizer, evaluate=True))
    else:
        eval_dataset = ListDataset([])
    if training_args.do_predict:
        predict_dataset = ListDataset(load_and_cache_reverse_examples_eval(model_args, tokenizer, evaluate=True))
    else:
        predict_dataset = ListDataset([])

    # Training
    if training_args.do_train:
        # Initialize our Trainer
        trainer = GenerationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=partial(generation_collate_fn, tokenizer=tokenizer),
            # prediction_loss_only=True
            compute_metrics=build_compute_metrics_fn(),
        )

        trainer.train(
            resume_from_checkpoint=None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # prediction
    eval_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")

        result = run_prediction(training_args, model_args, predict_dataset, model, tokenizer, output_prediction=True)
        # if trainer.is_world_process_zero():
        logger.info("***** Test results *****")
        for key, value in result.items():
            logger.info("  %s = %s", key, value)

        eval_results.update(result)
    return eval_results


if __name__ == "__main__":
    main()

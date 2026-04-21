import torch
import os
from constants import huggingface_home

os.environ['HF_HOME'] = huggingface_home

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, T5ForConditionalGeneration, BertConfig
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict
from pathlib import Path
from constants import HF_ACCESS_TOKEN
from pronouns import pronoun_mapping
from prompt import prompt_model
from minicons import scorer
import csv
import sys


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_mask_token(model):
    if 'roberta' in model.config._name_or_path:
        return '<mask>'
    return '[MASK]'

def get_decoder_loading_kwargs(model_name, use_low_cpu_mem_usage=False):
    model_name_lower = model_name.lower()
    loading_kwargs = {
        'token': HF_ACCESS_TOKEN,
        'torch_dtype': torch.float16,
    }

    if use_low_cpu_mem_usage:
        loading_kwargs['low_cpu_mem_usage'] = True

    if 'awq' in model_name_lower:
        if not torch.cuda.is_available():
            raise ValueError(f'AWQ model {model_name} requires a CUDA device.')
        # AWQ models do not support CPU or disk entries in the inferred device map.
        loading_kwargs['device_map'] = {'': torch.cuda.current_device()}
    else:
        loading_kwargs['device_map'] = 'auto'

    return loading_kwargs

def get_model(model_name, model_type):
    if model_type == 'encoder':
        if 'mosaic-bert' in model_name:
            config = BertConfig.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True, torch_dtype='auto').to(device)
        else:
            model = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype='auto').to(device)
    elif model_type == 'decoder':
        if any([s in model_name.lower() for s in ['66b', '70b']]):
            model = AutoModelForCausalLM.from_pretrained(
                    model_name, **get_decoder_loading_kwargs(model_name, use_low_cpu_mem_usage=True))
        elif '30b' in model_name:
            model = AutoModelForCausalLM.from_pretrained(
                    model_name, **get_decoder_loading_kwargs(model_name))
        elif any([s in model_name for s in ['12b', '13b']]):
            model = AutoModelForCausalLM.from_pretrained(
                    model_name, **get_decoder_loading_kwargs(model_name))
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_ACCESS_TOKEN,
                    torch_dtype='auto').to(device)
    elif model_type == 'enc-dec':
        model = T5ForConditionalGeneration.from_pretrained(model_name,
                torch_dtype=torch.float16, device_map='auto')
    else:
        raise ValueError('unsupported model type!')
    return model

def get_tokenizer(model_name):
    if 'mosaic-bert' in model_name:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_ACCESS_TOKEN)
    return tokenizer

# ordered by number of parameters
models = [
    #('microsoft/mdeberta-v3-base', 'encoder'), # 86M, multilingual
    #('microsoft/deberta-v3-base', 'encoder'), # 86M, English
    #('deepset/gelectra-base', 'encoder'), # 110M, German
    #('bert-base-uncased', 'encoder'),  # 110M, English
    ('deepset/gbert-base', 'encoder'), # 111M, German
    #('roberta-base', 'encoder'),  # 125M, English
    ('google-bert/bert-base-multilingual-cased', 'encoder'),  # 179M, multilingual
    ('FacebookAI/xlm-roberta-base', 'encoder'),  # 279, multilingual
    #('deepset/gelectra-large', 'encoder'),  # 336M, German
    ('deepset/gbert-large', 'encoder'), # 337M, German
    #('TechxGenus/Meta-Llama-3-8B-Instruct-AWQ', 'decoder'), # 1.98B, English
    ('mayflowergmbh/Llama-3-SauerkrautLM-8b-Instruct-AWQ', 'decoder'),  # 1.98B, German
    #('LeoLM/leo-hessianai-7b-chat', 'decoder'),  # 7B, German
    #('LeoLM/leo-hessianai-7b-chat-bilingual', 'decoder'),  # 7B, multilingual
    #('TechxGenus/Meta-Llama-3-70B-Instruct-AWQ', 'decoder'), # 11.3B, English
    ('tresiwalde/Llama-3-SauerkrautLM-70b-Instruct-AWQ', 'decoder'), # 11.3B, German
    ('casperhansen/llama-3.3-70b-instruct-awq', 'decoder'), # 70B, multilingual
    #('LeoLM/leo-hessianai-13b-chat', 'decoder'), # 13B, German
    #('LeoLM/leo-hessianai-70b-chat', 'decoder'), # 70B, German
]

def get_encoder_log_probs(sentence, pronoun_type, pronouns, mlm_scorer):
    log_prob_dict = {}
    for p in pronouns:
        verbalized = sentence.replace(pronoun_type, p)
        log_prob_dict[p] = mlm_scorer.sequence_score(verbalized,
                                                     reduction = lambda x: x.mean(0).item(),
                                                     PLL_metric='within_word_l2r')[0]
    return log_prob_dict

def get_decoder_log_probs(sentence, pronoun_type, pronouns, tokenizer, model):
    log_prob_dict = {}
    for p in pronouns:
        verbalized = sentence.replace(pronoun_type, p)
        input_ids = tokenizer(verbalized, return_tensors='pt').input_ids.to(device)
        outputs = model(input_ids).logits.detach()
        out_logits = outputs[0]
        log_probs = F.log_softmax(out_logits, dim=1) # convert to log probs by doing a log softmax
        log_prob_sum = 0.0
        token_count = 0
        for i in range(1, input_ids.shape[1]): # iterate over every input token position excluding BOS
            lp = log_probs[i-1, input_ids[0, i]] # logprob at the k-1-th position of the kth token in the input
            log_prob_sum += lp.item()
            token_count += 1
        log_prob_dict[p] = log_prob_sum/token_count
    return log_prob_dict

def construct_model_file_map(input_files):
    model_file_map = defaultdict(list)
    for data_file in input_files:
        print(data_file)
        # make directory for results
        stem = Path(data_file).stem
        folder = Path(stem)
        folder.mkdir(exist_ok=True)
        for MODEL, model_type in models:
            out_file = Path(folder / f"{MODEL.replace('/', '_')}.tsv")
            prompt_out_file = Path(folder / f"prompt_{MODEL.replace('/', '_')}.tsv")
            if out_file.exists() and prompt_out_file.exists():
                continue
            if not out_file.exists() and 'chat' not in MODEL and 'flan' not in MODEL:
                model_file_map[MODEL].append((model_type, data_file, out_file))
            if not prompt_out_file.exists() and ('chat' in MODEL or 'flan' in MODEL):
                model_file_map[MODEL].append((model_type, data_file, prompt_out_file))
    return model_file_map

def main():
    assert len(sys.argv) >= 2

    model_file_map = construct_model_file_map(sys.argv[1:])

    for MODEL in model_file_map:
        print(f'loading {MODEL}')
        model_type = model_file_map[MODEL][0][0]
        model = get_model(MODEL, model_type)
        tokenizer = get_tokenizer(MODEL)
        model.eval() # disable dropout
        for model_type, data_file, out_file in model_file_map[MODEL]:
            is_prompt = 'prompt' in out_file.name
            print(out_file)
            header = [
                'sentence',
                'verbalized_token',
                'pronoun_type',
                'occupation',
                'participant',
                'word',
                'article_and_occupation'
            ]
            if not is_prompt:
                if model_type == 'encoder':
                    mlm_scorer = scorer.MaskedLMScorer(model, tokenizer=tokenizer, device=device)
                with open(out_file, 'w') as out_f:
                    with open(data_file) as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        pll_header = header + [f'p_{p}' for p in pronoun_mapping['$NOM_PRONOUN']]
                        if 'pronoun' in reader.fieldnames:
                            pll_header += ['pronoun']
                        out_f.write('\t'.join(pll_header) + '\n')

                        for row in reader:
                            pronouns = pronoun_mapping[row['pronoun_type']]
                            if model_type == 'encoder':
                                # sentence-level pseudo log probabilities with different pronouns
                                associations = get_encoder_log_probs(
                                        row['sentence'],
                                        row['pronoun_type'],
                                        pronouns,
                                        mlm_scorer
                                        )
                            else:
                                # sentence-level log probabilities with different pronouns
                                associations = get_decoder_log_probs(
                                        row['sentence'],
                                        row['pronoun_type'],
                                        pronouns,
                                        tokenizer,
                                        model
                                        )
                            verbalized_token = sorted(associations.items(), key=lambda x: x[1], reverse=True)[0][0]

                            data = [
                                row['sentence'],
                                verbalized_token,
                                row['pronoun_type'],
                                row['occupation'],
                                row['participant'],
                                row['word'],
                                row['article_and_occupation']
                            ]
                            data += [f'{associations[pronouns[n]]}' for n in range(len(pronouns))]
                            if 'pronoun' in reader.fieldnames:
                                data += [row['pronoun']]
                            out_f.write('\t'.join(data) + '\n')
            elif is_prompt:
                with open(out_file, 'w', encoding='utf-8') as prompt_out_f:
                    with open(data_file) as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        prompt_header = [
                            'sentence',
                            'generation',
                            'pronoun_type',
                            'occupation',
                            'participant',
                            'word',
                            'article_and_occupation',
                            'prompt'
                        ]
                        if 'pronoun' in reader.fieldnames:
                            prompt_header += ['pronoun']
                        prompt_out_f.write('\t'.join(prompt_header) + '\n')

                        for row in reader:
                            pronouns = pronoun_mapping[row['pronoun_type']]
                            for prompt, generation in prompt_model(row['sentence'], row['pronoun_type'], pronouns, row['word'], row['article_and_occupation'], tokenizer, model, model_type, MODEL):
                                data = [
                                    row['sentence'],
                                    generation,
                                    row['pronoun_type'],
                                    row['occupation'],
                                    row['participant'],
                                    row['word'],
                                    row['article_and_occupation'],
                                    str(prompt)
                                ]
                                if 'pronoun' in reader.fieldnames:
                                    data += [row['pronoun']]
                                prompt_out_f.write('\t'.join(data) + '\n')

if __name__ == '__main__':
    main()

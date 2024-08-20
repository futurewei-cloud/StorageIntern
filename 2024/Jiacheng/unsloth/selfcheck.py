

import spacy
import os
PATH = '/data/changjli/hugging_cache'
os.environ['HF_HOME'] = PATH
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
from transformers import logging
logging.set_verbosity_error()
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LongformerTokenizer, LongformerForMultipleChoice, LongformerForSequenceClassification
import re





class SelfCheckLLMPrompt:
    """
    SelfCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via open-source LLM prompting
    """

    def __init__(
            self,
            model_name: str = None,
            device=None
    ):
        model = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model, token="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")
        self.model = AutoModelForCausalLM.from_pretrained(model, token="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", torch_dtype="auto")
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        self.prompt_template = "Thought: {thought}\n\nAction: {action}\n\nIs the action supported by the thought above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"SelfCheck-LLMPrompt ({model}) initialized to device {device}")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
    def predict(
            self,
            sentences: List[str],
            sampled_passages: List[str],
            verbose: bool = False,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """

        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ")

                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,
                    do_sample=False,  # hf's default for Llama2 is True
                )
                output_text = self.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                generate_text = output_text.replace(prompt, "")
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    @torch.no_grad()
    def predict_json(self, json_file, type="os"):
        items = []
        with open(json_file, 'r') as file:
            for line in file:
                items.append(json.loads(line))

        if type == "os":
            start_index = 6
            think_pattern = r"Think: (.+?)\n"
            act_pattern = r"Act:\s*(.+)"
        else:
            start_index = 0
            think_pattern = r"Think: (.+?)\n"
            act_pattern = r"Act:\s*(.+)"

        for conversatrions in items:

            for conv in conversatrions[start_index:]:
                if conv['from'] == 'gpt':
                    think_match = re.search(think_pattern, conv['value'])
                    act_match = re.search(act_pattern, conv['value'])

                    think = think_match.group(1)
                    act = act_match.group(1)

                    if act =="bash":

                        content_pattern = r"```bash\n(.*?)\n```"
                        content = re.findall(content_pattern, conv['value'], re.DOTALL)
                        content = "\n\n".join(content)
                        act = act + "\n" + content

                    elif act == "finish":
                        act = "commit"

                    elif act.startswith("answer"):
                         act = act

                    prompt = self.prompt_template.format(thought=think, action=act)
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                    generate_ids = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=5,
                        do_sample=False,  # hf's default for Llama2 is True
                    )
                    output_text = self.tokenizer.batch_decode(
                        generate_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                    generate_text = output_text.replace(prompt, "")
                    score_ = self.text_postprocessing(generate_text)
                    #scores[sent_i, sample_i] = score_
                else:
                    pass










    def text_postprocessing(
            self,
            text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        # tested on Llama-2-chat (7B, 13B) --- this code has 100% coverage on wikibio gpt3 generated data
        # however it may not work with other datasets, or LLMs
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]



if __name__ == '__main__':
     selfcheck = SelfCheckLLMPrompt(model_name="mistralai/Mistral-7B-Instruct-v0.2", device="cuda:0")

     selfcheck.predict_json("/home/changjli/ETO/data/os_poison_mniddle.json", type="os")
from src.utils import *
from tqdm import tqdm
from nltk import tokenize

import torch
from torch.utils.data import DataLoader, Dataset

import datasets
from datasets import ClassLabel, load_dataset

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForTokenClassification,
)

model_path = "douy/deberta-v3-large-self-disclosure-detection"


class DisclosureDataset(Dataset):
    def __init__(self, inputs, tokenizer, tokenize_and_align_labels_function):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.tokenize_and_align_labels_function = tokenize_and_align_labels_function

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        words = self.inputs[idx]
        tokenized_inputs = self.tokenize_and_align_labels_function(words)
        return tokenized_inputs


def collate_spans(tokens_with_labels):
    collated_spans = []
    current_span = []
    current_label = None

    for token, label in tokens_with_labels:
        # If the label is O, we don't want to include it in spans
        # if label == 'O':
        #     if current_span:
        #         collated_spans.append((current_label, ' '.join(current_span)))
        #         current_span = []
        #     current_label = None
        #     continue

        # Extract the label prefix (B or I) and actual label
        label_type = label.split("-", 1)[-1]  # Split to ignore 'B-' or 'I-'

        # If it's a new label, start a new span
        if not current_label or current_label != label_type:
            if current_span:
                collated_spans.append((current_label, " ".join(current_span)))
            current_label = label_type
            current_span = [token]
        else:
            # If it's the same label, continue the span
            current_span.append(token)

    # Append the last span if any
    if current_span:
        collated_spans.append((current_label, " ".join(current_span)))

    return collated_spans

def generate_span_prediction_pipeline():
    config = AutoConfig.from_pretrained(
        model_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_path, config=config, device_map="cuda:0"
    ).eval()

    label2id = config.label2id
    id2label = config.id2label

    def tokenize_and_align_labels(words):
        tokenized_inputs = tokenizer(
            words,
            padding=False,
            is_split_into_words=True,
        )

        # we use ("O") for all the labels
        word_ids = tokenized_inputs.word_ids(0)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label2id["O"])
            # For the other tokens in a word, we set the label to -100
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs

    def batch_process(inputs):
        data_collator = DataCollatorForTokenClassification(tokenizer)

        disclosure_dataset = DisclosureDataset(inputs, tokenizer, tokenize_and_align_labels)

        dataloader = DataLoader(disclosure_dataset, collate_fn=data_collator, batch_size=8)

        total_predictions = []
        for step, batch in enumerate(tqdm(dataloader, desc="Identifying disclosure spans")):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.inference_mode():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(-1)
            labels = batch["labels"]

            predictions = predictions.cpu().tolist()
            labels = labels.cpu().tolist()

            true_predictions = []
            for i, label in enumerate(labels):
                true_pred = []
                for j, m in enumerate(label):
                    if m != -100:
                        true_pred.append(id2label[predictions[i][j]])
                true_predictions.append(true_pred)
            total_predictions.extend(true_predictions)
        return total_predictions
    return batch_process



input_format = """Your task is to generalize the given 'disclosure span' in the sentence. Provide three diverse generalized spans that convey similar meaning but remove any overly specific or sensitive information.

Remember the following criteria:
* Only the disclosure span should be generalized; the rest of the sentence should remain intact.
* Generalized spans should be diverse but should all retain the essence of the original span.
* Make sure the generalized span fits seamlessly into the original sentence, maintaining proper syntax and grammar.
* Provide three diverse generalized alternatives in a JSON format like this: {{"span 1": "xxx", "span 2": "xxx", "span 3": "xxx"}}.

Sentence: "{sentence}"
Disclosure Span to Revise: "{span}"
Generalized Spans:"""


def generate_abstraction_pipeline():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    peft_model_id = "douy/Llama-2-7B-lora-instruction-ft-abstraction-three-span"
    config = PeftConfig.from_pretrained(peft_model_id)

    tokenizer = AutoTokenizer.from_pretrained(peft_model_id, padding_side="left")

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, device_map="cuda:0"
    )

    model.resize_token_embeddings(32008)

    model = PeftModel.from_pretrained(model, peft_model_id)

    model.generation_config.top_p = 1
    model.generation_config.temperature = 0.7

    model.eval()

    # put into your data
    # def batch_process(sentences, spans, batch_size=16):
    def batch_process(sentences, spans, batch_size=4):
        generated_texts = []

        # Split sentences and spans into batches
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_sentences = sentences[i : i + batch_size]
            batch_spans = spans[i : i + batch_size]

            batch_text = [
                input_format.format(sentence=sentence, span=span)
                for sentence, span in zip(batch_sentences, batch_spans)
            ]

            inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to(0)

            outputs = model.generate(
                **inputs,
                # max_new_tokens=1024,
                max_new_tokens=128,
                do_sample=True,
                top_p=1,
                temperature=0.7,
                num_return_sequences=1,
            )
            input_length = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[:, input_length:]
            generated_text = tokenizer.batch_decode(
                generated_tokens.detach().cpu().numpy(), skip_special_tokens=True
            )
            generated_texts.extend(generated_text)
        return generated_texts

    return batch_process



def process(
    step_conf,
    global_conf,
    dataset,
    access_func,
    output_path,
    print_and_write_results,
    print_and_write_results_json,
    output_dataset_key,
    gpt_method,
    method_name_config,
    dataset_name,
    **kwargs,
):

    if "wildchat" in dataset_name:
        adv_inf_instr = "Let us play a guessing game. The following is a synthetic patient record. Your task is to guess the patient's attributes including Income, Education, Sex, Relationship Status, Location, Place of Birth, and Age?"
        data_type_sanitization = "medical record from a user"
        data_type_word = "Record"
    else:
        adv_inf_instr = "Let us play a guessing game. The following is the a conversation between a user and an AI chatbot. Your task is to guess the user's attributes including Income, Education, Sex, Relationship Status, Location, Place of Birth, and Age?"
        data_type_sanitization = "conversation between a user and an AI chatbot"
        data_type_word = "Conversation"

    records = access_func(dataset)
    sentences_per_record = [tokenize.sent_tokenize(record) for record in records]

    sentences = [sentence for record in sentences_per_record for sentence in record]

    inputs = [sentence.split() for sentence in sentences]

    span_prediction_pipeline = generate_span_prediction_pipeline()
    total_predictions = span_prediction_pipeline(inputs)

    span_sentences, spans = [], []
    for input, total_prediction in zip(inputs, total_predictions):
        original_input = " ".join(input)
        collated_spans = collate_spans(list(zip(input, total_prediction)))

        for label, span in collated_spans:
            if label != "O":
                span_sentences.append(original_input)
                spans.append(span)

    del span_prediction_pipeline
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    abstration_process = generate_abstraction_pipeline()
    generalized_spans = abstration_process(span_sentences, spans)
    new_generalized_spans = []
    for generalized_span in generalized_spans:
        try:
            generalized_span = json.loads(generalized_span)["span 1"]
        except:
            try:
                generalized_span = (
                    generalized_span.split("span 1")[-1]
                    .split("span 2")[0][4:-4]
                    .strip()
                )
            except:
                generalized_span = (
                    generalized_span.split("span 1")[-1].split("span 2")[0].strip()
                )
        new_generalized_spans.append(generalized_span)

    generalized_spans = new_generalized_spans
    idx = 0

    sanitized_sentences = []
    for input, total_prediction in zip(inputs, total_predictions):
        # original_input = " ".join(input)
        sentence_spans = []
        collated_spans = collate_spans(list(zip(input, total_prediction)))

        for label, span in collated_spans:
            if label != "O":
                generalized_span = generalized_spans[idx]

                idx += 1
                sentence_spans.append(generalized_span)
            else:
                sentence_spans.append(span)
        sentence = " ".join(sentence_spans)
        sanitized_sentences.append(sentence)

    assert len(sanitized_sentences) == len(sentences)
    sanitized_records = []
    idx = 0
    for sentence_per_record in sentences_per_record:
        sanitized_records.append(
            " ".join(sanitized_sentences[idx : idx + len(sentence_per_record)])
        )
        idx += len(sentence_per_record)
    dataset = dataset.add_column(output_dataset_key, sanitized_records)

    set_access_key(dataset, "sanitized_document", output_dataset_key, global_conf)

    return dataset

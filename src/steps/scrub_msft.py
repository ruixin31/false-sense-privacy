from src.utils import *
import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


language_key = os.environ.get("LANGUAGE_KEY")
language_endpoint = os.environ.get("LANGUAGE_ENDPOINT")


# Authenticate the client using your key and endpoint
def authenticate_client():
    ta_credential = AzureKeyCredential(language_key)
    text_analytics_client = TextAnalyticsClient(
        endpoint=language_endpoint, credential=ta_credential
    )
    return text_analytics_client


client = authenticate_client()


# Example method for detecting sensitive information (PII) from text
def pii_recognition_example(dataset, access_func, output_key):
    documents = access_func(dataset)

    # exisitng_documents = dataset[output_key]
    # if None not in exisitng_documents:
    #     return dataset
    # documents = dataset["context"]

    response = client.recognize_pii_entities(documents, language="en")
    outputs = []
    # result = [doc for doc in response if not doc.is_error]
    for doc in response:
        if not doc.is_error:
            output = {
                output_key: doc.redacted_text,
                "entities": [
                    {
                        "text": entity.text,
                        "category": entity.category,
                        "confidence_score": entity.confidence_score,
                        "offset": entity.offset,
                        "length": entity.length,
                    }
                    for entity in doc.entities
                ],
            }
        else:
            output = {
                output_key: "",
                "entities": [],
            }
        outputs.append(output)

    first_sample = outputs[0]
    for key in first_sample:
        dataset[key] = [sample[key] for sample in outputs]

    return dataset


# In some sense this is much "safer"-- that is, the wrong execution sequence was eliminated in the previous answer phase
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
    dataset = dataset.map(
        pii_recognition_example,
        batched=True,
        batch_size=5,
        fn_kwargs={"access_func": access_func, "output_key": output_dataset_key},
        desc="Inference",
    )

    set_access_key(dataset, "sanitized_document", output_dataset_key, global_conf)
    return dataset

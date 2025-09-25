from src.utils import *
from rouge_score import rouge_scorer
import numpy as np
from src.load_dataset import load_dataset_from_config
from functools import partial
import re

def post_process_facts(responses):
    # Regular expression to match curly braces along with their contents
    # Find all matches of the pattern in the text
    patterns = [
        # re.compile(r"claim ?.?: ?(.+)", re.IGNORECASE | re.MULTILINE),
        # re.compile(r"^\d+: ?(.+)", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\d+\. ?(.+)", re.IGNORECASE | re.MULTILINE),
    ]

    try:
        for pattern in patterns:
            matches = re.findall(pattern, responses)
            subclaims = matches
            if subclaims:
                break
        # if len(subclaims) == 0:
        #     breakpoint()
        #     raise
        # subclaims = [match[0] for match in matches]
        # subclaims = json.loads(subclaims)
        if not subclaims:
            subclaims = responses.split("\n")[1:]
            # subclaims = map(lambda x: x.split(":")[-1], subclaims)
        subclaims = list(filter(None, subclaims))
        # breakpoint()
        # if len(subclaims) < len(list(filter(None, responses.split("\n")))) - 2:

        #     print(responses)
        #     print(subclaims)
        #     breakpoint()
    except Exception as ex:
        print(ex)
        print("Failed to parse as jsonl")
        print(responses)
        subclaims = None
    # print('=====')
    # print(responses)
    # print('=====')
    # print(subclaims)
    # breakpoint()
    return subclaims




def process(
    step_conf,
    global_conf,
    dataset,
    access_func,
    output_path,
    print_and_write_results,
    output_dataset_key,
    gpt_method,
    method_name_config,
    dataset_name,
    **kwargs,
):
    # if method_name_config.get('baseline', False):
    #     output_dataset_key = 'facts-baseline'


    if 'wildchat' in dataset_name:
        user_summarization_instruction = "Read the following conversation between a user and an AI chatbot. Summarize the user's input, maintaining as much of the original phrasing as possible. When contextually appropriate, include the chatbot's response to ensure the conversation's essence is captured."
        icl_samples = f"""{user_summarization_instruction}

USER: According to UK law, how much time does an agency need to return a deposit in to a tenant in the uk?
CHATBOT: According to UK law, the time frame for returning a deposit to a tenant depends on whether the deposit is protected in a government-backed tenancy deposit scheme (TDP) or not. If the deposit is protected in a TDP, the agency must return the deposit within 10 days of reaching an agreement with the tenant on the amount to be returned. However, if the deposit is not protected in a TDP, the agency must return the deposit within 10 days of the end of the tenancy or within 10 days of the landlord and tenant agreeing on the amount to be returned, whichever is later. It is worth noting that failure to protect a deposit in a TDP can result in financial penalties for the agency.
USER: If it is protected in a tenancy deposit scheme what is the time?
CHATBOT: If the deposit is protected in a government-backed tenancy deposit scheme (TDP), the agency must return the deposit within 10 days of reaching an agreement with the tenant on the amount to be returned.
USER: What would the agreement be?
CHATBOT: The agreement regarding the amount of the deposit to be returned typically occurs between the landlord/agency and the tenant after any deductions have been made. This agreement could be reached through mutual negotiation or, in case of a dispute, through an alternative dispute resolution (ADR) process provided by the TDP scheme. The agreement would include the final amount to be returned to the tenant, considering any valid deductions for damages, outstanding rent, or other permissible charges.

[INST]
The user first asked how much time an agency has to return a deposit to a tenant according to UK law. The chatbot explained that if the deposit is protected in a government-backed tenancy deposit scheme (TDP), it must be returned within 10 days of reaching an agreement on the amount to be returned. If not protected, it must also be returned within 10 days of the end of the tenancy or the agreement, whichever is later, with potential financial penalties for non-compliance. The user then confirmed the time frame for a deposit protected in a TDP, which the chatbot reiterated as 10 days after reaching an agreement. Finally, the user inquired about the nature of the agreement, and the chatbot clarified that the agreement on the returned amount usually follows negotiations or an alternative dispute resolution (ADR) process and includes any deductions for damages, outstanding rent, or permissible charges.
[INST]"""
        prompt = f"""{icl_samples}

{user_summarization_instruction}

{{}}
"""
        prompt = Prompt(
            prompt,
        )

        intermediate_key = "summarized_user_input"
        dataset = prompt.apply(dataset, gpt_method, access_func, intermediate_key)
        access_func = intermediate_key


    instruction = 'Please breakdown the following input into a set of atomic, independent claims, and return each of the claim in a new line.'

    icl_samples = f"""{instruction}

A 62-year-old woman presents for a regular check-up. She complains of lightheadedness and palpitations which occur episodically. Past medical history is significant for a myocardial infarction 6 months ago and NYHA class II chronic heart failure. She also was diagnosed with grade I arterial hypertension 4 years ago. Current medications are aspirin 81 mg, atorvastatin 10 mg, enalapril 10 mg, and metoprolol 200 mg daily. Her vital signs are a blood pressure of 135/90 mm Hg, a heart rate of 125/min, a respiratory rate of 14/min, and a temperature of 36.5°C (97.7°F). Cardiopulmonary examination is significant for irregular heart rhythm and decreased S1 intensity. ECG is obtained and is shown in the picture (see image). Echocardiography shows a left ventricular ejection fraction of 39%.

[INST]
Here are the atomic, independent claims broken down into individual lines:

1. A 62-year-old woman presents for a regular check-up.
2. She complains of lightheadedness.
3. She complains of palpitations which occur episodically.
4. Her past medical history is significant for a myocardial infarction 6 months ago.
5. Her past medical history is significant for NYHA class II chronic heart failure.
6. Her past medical history is significant for grade I arterial hypertension 4 years ago.
7. She is currently taking aspirin 81 mg daily.
8. She is currently taking atorvastatin 10 mg daily.
9. She is currently taking enalapril 10 mg daily.
10. She is currently taking metoprolol 200 mg daily.
11. Her blood pressure is 135/90 mm Hg.
12. Her heart rate is 125/min.
13. Her respiratory rate is 14/min.
14. Her temperature is 36.5°C (97.7°F).
15. Cardiopulmonary examination is significant for irregular heart rhythm.
16. Cardiopulmonary examination is significant for decreased S1 intensity.
17. An ECG is obtained.
18. The ECG is shown in the picture (see image).
19. Echocardiography shows a left ventricular ejection fraction of 39%.
[INST]"""


    prompt = f"""

{icl_samples}

{instruction}

{{}}
"""
    prompt = Prompt(
        prompt,
        post_process_func=post_process_facts,
    )

    dataset = prompt.apply(dataset, gpt_method, access_func, output_dataset_key)

    if method_name_config.get('baseline', False):
        set_access_key(dataset, "original_facts", output_dataset_key, global_conf)
    else:
        set_access_key(dataset, "sanitized_facts", output_dataset_key, global_conf)

    return dataset
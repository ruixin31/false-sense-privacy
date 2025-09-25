from typing import DefaultDict
import tiktoken
from tqdm import tqdm
from rouge_score import rouge_scorer
from src.utils import PromptWithCache, proper_slice_array_by_range
from collections import defaultdict
from statistics import mode
from pathlib import Path
from src.gpt import LM
import re

BATCH_SIZE = 10
BIN_ID = "queries"

calls = 0
errors = 0
def post_process(responses, prompt, expected_len=BATCH_SIZE):
    global calls, errors
    calls += 1
    # pattern = re.compile(r"(\d+)\..*score: (.*)$", re.IGNORECASE | re.MULTILINE)
    pattern = re.compile(
        r"Claim (\d+) Similarity Rating: (\d)", re.IGNORECASE | re.MULTILINE
    )
    pattern_3 = re.compile(
        r"Claim Similarity Rating: (\d)", re.IGNORECASE | re.MULTILINE
    )
    pattern_2 = re.compile(
        r"Rating.{0,5}?(\d)", re.IGNORECASE | re.MULTILINE
    )
    scores_dict: DefaultDict[int, list[int]] = defaultdict(list)

    for response in responses:
        found = False
        if not found:
            matches = re.findall(pattern, response)
            # if len(matches) > 1:
            #     print(prompt, responses)
            for i in matches:
                try:
                    idx = int(i[0]) - 1
                    score = int(i[1].strip())
                    scores_dict[idx].append(score)
                    found = True
                except:
                    pass
        if not found:
            matches = re.findall(pattern_3, response)
            if len(matches) == expected_len:
                try:
                    for idx, score in enumerate(matches):
                        scores_dict[idx].append(int(score.strip()))
                    found = True
                except:
                    pass
        if not found:
            matches = re.findall(pattern_2, response)
            if len(matches) == expected_len:
                try:
                    for idx, score in enumerate(matches):
                        scores_dict[idx].append(int(score.strip()))
                    found = True
                except:
                    pass
            # else:
            #     # print(len(matches), prompt, responses)
            #     print(len(matches), response)
    ### take the mode of the scores

    if len(scores_dict.keys()) == 0:
        # print("No scores found")
        # print(responses)
        # breakpoint()
        errors += 1
        return []

    scores: list[float] = []
    for idx in range(max(scores_dict.keys()) + 1):
        if scores_dict[idx]:
            score = mode(scores_dict[idx])
            if score < 1 or score > 3:
                print(score)
                score = 1
            score = (score - 1) / 2
            scores.append(score)
        else:
            scores.append(-1)
            # scores.append(None)

    # if scores and scores[0] == 1:
    #     # pass
    #     print(prompt)
    #     list(map(print, responses))
    #     breakpoint()
    if len(scores) != expected_len:
        errors += 1
    return scores

def ensure_scores(scores, expected_len):
    if len(scores) == 0:
        return [-1] * expected_len
    if len(scores) > expected_len:
        scores = scores[:expected_len]
    else:
        scores += [-1] * (expected_len - len(scores))

    return scores




icl_wildchat = """**Input**:
- Claim 1: She has lived for 25 years.
- Claim 2: She is an adult.
- Claim 3: She does not have a fever.
Original text: A 25-year-old female has a body temperature of 38.5 degrees Celsius.

[INST]
**Output**:
- Explanation: Claim 1 conveys the same information as Claim 1 but in different wording.
- Claim 1 Similarity Rating: 1 (Same Information)
- Explanation: Claim 2 states that she is an adult, which is implied given she is 25 years old. This closely aligns with the information in the original text, though it is not explicitly stated.
- Claim 2 Similarity Rating: 2 (Different but Similar Information)
- Explanation: Claim 3 states that she does not have a fever, which contradicts the original text, where her body temperature is 38.5 degrees Celsius, indicating a fever.
- Claim 3 Similarity Rating: 3 (Unsupported)
[INST]

**Input**:
- Claim 1: The user's girlfriend posted a video with a love song and another guy.
- Claim 2: The user is advised to not communicate with his girlfriend.
- Claim 3: The chatbot suggests communicating openly about feelings and concerns is important.
Original text: USER: my girlfriend posted a video with a boy and she tittled it #inlove with a love song and i stoped texting her am i in the wrong
CHATBOT: Being in the right or wrong in a situation like this can be subjective, based mainly on the boundaries and expectations that you and your girlfriend have established in your relationship. If this action made you uncomfortable, it's completely valid to feel this way. However, it'll be beneficial to communicate openly about your feelings and concerns with her. Remember that assumptions can lead to misunderstandings, so it would be best to ask her directly about the content she posted. Communication is key in any relationship. So, no, you're not in the wrong for feeling uneasy, but you should talk to her about it rather than stopping communication completely.

[INST]
**Output**:
- Explanation: Claim 1 includes the same specific details from the original text about the girlfriend's video post with a boy and a love song.
- Claim 1 Similarity Rating: 1 (Same Information)
- Explanation: Claim 2 completely contradicts the advice given by the chatbot. In fact, the chatbot is encouraging communication rather than advising against it.
- Claim 2 Similarity Rating: 3 (Unsupported)
- Explanation: Claim 3 mirrors the advice given by the chatbot exactly, emphasizing the importance of open and direct communication.
- Claim 3 Similarity Rating: 1 (Same Information)
[INST]

**Input**:
- Claim 1: She is 25 years old.
- Claim 2: She is a cook.
Original text: A 20-something-years-old female has a body temperature of 38.5 degrees Celsius.

[INST]
**Output**:
- Explanation: The passage only mentions that she is "20-something-years-old," which indicates she is between 20-29 years old. While 25 would fall within this range, the specific age of 25 is not explicitly stated in the passage. The passage provides an approximate age range rather than the exact age claimed.
- Claim 1 Similarity Rating: 2 (Different but Similar Information)
- Explanation: The passage contains no information about the woman's occupation. The only details provided are her approximate age range, gender, and body temperature. There is no mention of her being a cook or any information about her profession at all.
- Claim 2 Similarity Rating: 3 (Unsupported)
[INST]

**Input**:
- Claim 1: The user asked to write a thesis about a major conflict in Arthur Miller's play The Crucible.
- Claim 2: The conflict is between telling the truth and risking death or lying to protect one's life.
- Claim 3: The chatbot provided a thesis statement and analyzed the conflict through the characters of John Proctor and Mary Warren.
- Claim 4: The user requested a detailed analysis of at least two scenes for each character, highlighting the tension between truth and survival.
- Claim 5: John Proctor's character embodies the conflict between truth and survival.
- Claim 6: Initially, Proctor's deception about his affair with Abigail Williams weakens his credibility.
- Claim 7: Proctor ultimately decides to confess the truth and risk his own life to expose the witchcraft hysteria.
- Claim 8: Two scenes were highlighted for John Proctor: his confrontation with the court in Act III, where he exposes his affair with Abigail, and his final scene, where he rejects the option to sign a false confession and preserve his life.
- Claim 9: Proctor's unwavering commitment to truth, even at the cost of his life, emphasizes Miller's assertion that truth must prevail over personal safety.
- Claim 10: Mary Warren's character struggles with the choice between protecting herself or exposing the lies behind the Salem witch trials.
Original text: USER: Write a letter to complain about the aggressive behavior of the interface manager when started to explain the site activities
CHATBOT: [Your Name] [Your Address] [City, State, ZIP] [Email Address] [Phone Number] [Date] [Manager’s Name] [Manager’s Position] [Company Name] [Company Address] [City, State, ZIP] Dear [Manager’s Name], I hope this letter finds you well. I am writing to express my deep concern and disappointment regarding an incident that occurred during my recent interaction with the interface manager at your company. On [date], I had the opportunity to meet with the interface manager to understand the various site activities that your company offers. From the onset of the meeting, it became evident that the manager’s behavior was excessively aggressive and lacked the professionalism and courtesy that I expect from a representative of your company. As I inquired about various sections available on the site, the interface manager began to display impatience and responded with a condescending tone. Instead of providing clear explanations and willingly addressing my questions, they seemed overly eager to dismiss my inquiries and brush off any concerns I may have had. This confrontational approach left me feeling intimidated and frustrated, hindering my ability to gather the information I had initially intended to share. I firmly believe that a positive customer experience entails treating all individuals with respect and valuing their opinions. Unfortunately, the interface manager’s hostile behavior during our meeting contradicted these principles, leaving a negative impact on the way I perceive your company. As a loyal customer, I was disheartened by the lack of empathy and professionalism exhibited during our interaction. I understand that misunderstandings or miscommunications may occur in any business setting, but it is crucial to address these instances to ensure future improvements. I kindly request that appropriate action be taken to address the aggressive behavior of the interface manager. Training programs or coaching sessions focusing on effective communication and ensuring a customer-centric approach could greatly benefit the manager’s interaction with clients in the future. In addition, I urge your company to consider initiating a feedback mechanism that allows customers to securely report any negative experiences they may encounter. This would not only enable the company to rectify existing issues promptly but also create a sense of urgency and accountability in addressing any challenges that arise. I sincerely hope that steps will be taken to address this matter, as I have always held a positive opinion of your company and would like to remain a loyal customer. I firmly believe that by addressing this issue, you can create an environment where customers feel truly

[INST]
- Explanation: Claim 1 is unrelated to the content of the passage. The passage discusses a complaint letter about an interface manager's aggressive behavior, not a thesis about a conflict in "The Crucible."
- Claim 1 Similarity Rating: 3 (Unsupported)
- Explanation: Claim 2 is also unrelated to the passage. The passage does not mention any conflict regarding truth and survival, nor does it refer to "The Crucible."
- Claim 2 Similarity Rating: 3 (Unsupported)
- Explanation: Claim 3 discusses an analysis of a conflict in "The Crucible" through the characters, which is entirely unrelated to the passage about a complaint letter.
- Claim 3 Similarity Rating: 3 (Unsupported)
- Explanation: Claim 4 refers to a detailed analysis of scenes from "The Crucible," which is not supported by the passage context regarding a complaint letter.
- Claim 4 Similarity Rating: 3 (Unsupported)
- Explanation: Claim 5 discusses John Proctor's character in "The Crucible" embodying a conflict, which the passage about a complaint letter does not address.
- Claim 5 Similarity Rating: 3 (Unsupported)
- Explanation: Claim 6 mentions Proctor's deception about an affair, which is irrelevant to the passage, where there is no mention of "The Crucible" or its characters.
- Claim 6 Similarity Rating: 3 (Unsupported)
- Explanation: Claim 7 discusses Proctor's decision in "The Crucible," which is not addressed or mentioned in the passage about writing a complaint letter.
- Claim 7 Similarity Rating: 3 (Unsupported)
- Explanation: Claim 8 highlights specific scenes involving John Proctor, which is completely unrelated to the passage that focuses on a complaint about the interface manager.
- Claim 8 Similarity Rating: 3 (Unsupported)
- Explanation: Claim 9 focuses on Proctor's commitment to truth, which is not mentioned or relevant to the passage about the complaint letter.
- Claim 9 Similarity Rating: 3 (Unsupported)
- Explanation: Claim 10 refers to Mary Warren's struggle in "The Crucible," which has no connection to the passage content regarding the complaint about an interface manager's behavior.
- Claim 10 Similarity Rating: 3 (Unsupported)
[INST]
"""

icl_medqa = """**Input**:
- Claim 1: She is 25 years old.
- Claim 2: She is a cook.
Original text: A 20-something-years-old female has a body temperature of 38.5 degrees Celsius.

[INST]
**Output**:
- Explanation: The passage only mentions that she is "20-something-years-old," which indicates she is between 20-29 years old. While 25 would fall within this range, the specific age of 25 is not explicitly stated in the passage. The passage provides an approximate age range rather than the exact age claimed.
- Claim 1 Similarity Rating: 2 (Different but Similar Information)
- Explanation: The passage contains no information about the woman's occupation. The only details provided are her approximate age range, gender, and body temperature. There is no mention of her being a cook or any information about her profession at all.
- Claim 2 Similarity Rating: 3 (Unsupported)
[INST]

**Input**:
Claim 1: A scientist performed a memory experiment on college students.
Claim 2: The study was conducted on healthy students.
Original text: A researcher conducted a memory test on a group of university students.

[INST]
**Output**:
Explanation: The claim uses synonymous terms (researcher/scientist, university/college) and conveys exactly the same meaning as the original text.
Claim 1 Similarity Rating: 1 (Same Information)
Explanation: While "healthy" is not mentioned in the original text, it's a reasonable assumption for university students participating in a study, though not explicitly stated.
Claim 2 Similarity Rating: 2 (Different but Similar Information)
[INST]

**Input**:
Claim 1: The patient had headaches during the last week.
Claim 2: The patient experienced several headaches.
Claim 3: The patient had mild headaches.
Claim 4: The patient took aspirin for the headaches.
Original text: The patient reported experiencing intermittent headaches over the past week.

[INST]
**Output**:
Explanation: The claim is a direct paraphrase that maintains the same meaning as the original text, just using different words to express the same information.
Claim 1 Similarity Rating: 1 (Same Information)
Explanation: "Several" and "intermittent" effectively convey the same meaning in this context - multiple occurrences over time.
Claim 2 Similarity Rating: 1 (Same Information)
Explanation: While severity wasn't mentioned in the original text, it's a reasonable inference about the nature of intermittent headaches, though not explicitly stated.
Claim 3 Similarity Rating: 2 (Different but Similar Information)
Explanation: There is no mention of any treatment or medication in the original text. This is completely new information not supported by the passage.
Claim 4 Similarity Rating: 3 (Unsupported)
[INST]
"""


# icl_medqa = """**Input**:
# **Original text**:
# A 20-something-years-old female has a body temperature of 38.5 degrees Celsius.

# **Claims**:
# - Claim 1: She is 25 years old.
# - Claim 2: She is a cook.


# [INST]
# **Output**:
# - Explanation: The passage only mentions that she is "20-something-years-old," which indicates she is between 20-29 years old. While 25 would fall within this range, the specific age of 25 is not explicitly stated in the passage. The passage provides an approximate age range rather than the exact age claimed.
# - Claim 1 Similarity Rating: 2 (Different but Similar Information)
# - Explanation: The passage contains no information about the woman's occupation. The only details provided are her approximate age range, gender, and body temperature. There is no mention of her being a cook or any information about her profession at all.
# - Claim 2 Similarity Rating: 3 (Unsupported)
# [INST]

# **Input**:
# **Original text**:
# A 25-year-old female has a body temperature of 38.5 degrees Celsius.

# **Claims**:
# - Claim 1: She does not have a fever.

# [INST]
# **Output**:
# - Explanation: Claim 1 states that she does not have a fever, which contradicts the original text, where her body temperature is 38.5 degrees Celsius, indicating a fever.
# - Claim 1 Similarity Rating: 3 (Unsupported)
# [INST]
# """


def formulate_entailment_prompt(context, facts, icl):
    facts_prepared = "\n".join(
        [f"- Claim {count + 1}: {fact}" for count, fact in enumerate(facts)]
    )

# **Format**:
# **Input**:
# - Claim 1: [Claim 1]
# - Claim 2: [Claim 2]
# Original text: [Original text]

# **Output**:
# - Explanation: [Brief explanation justifying the rating]
# - Claim 1 Similarity Rating: [Rating (1-3)]
# - Explanation: [Brief explanation justifying the rating]
# - Claim 2 Similarity Rating: [Rating (1-3)]

    prompt = f"""**Task**:
For each of the following claims, rate how well the provided passage supports it on a scale of 1 to 3, where:
- **1** means “Same Information,”
- **2** means “Different but Similar Information,” and
- **3** means “Unsupported.”

Use the rubric below to guide your evaluation and include a brief explanation for your rating to justify your decision. Focus on the content and facts being discussed, ignoring wording differences. Evaluate each claim separately and do not group them together in your output.

**Rubric**:
1. *Same Information*: The claim exactly matches the corresponding information in the passage. The claim could be a direct quote from the passage or a paraphrase that captures the same information.
2. *Different but Similar Information*: The claim is not identical to the passage but captures the same or closely related information. For example, the claim could be an abstraction of the corresponding information in the passage, or it could have minor differences that do not significantly change its overall meaning.
3. *Unsupported*: The claim is not supported by the passage. It could either be not discussed at all in the passage or inconsistent with the passage.

{icl}

**Input**:
{facts_prepared}
original text: {context}
"""
    return prompt


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
    # Privacy-specific arguments
    facts_key_orig,
    facts_key_sanitized,
    context_key_orig,
    context_key_sanitized,
    range_maps,
    ranges,
    get_top_article_idx,
    dataset_name,
    cfc_range,
    **kwargs,
):
    enc = tiktoken.get_encoding("o200k_base")
    do_precision = global_conf.get("do_precision", False)
    # Gather possible chatgpt queries

    result = {}
    exp_id = "fact_privacy"
    
    # icl = icl_wildchat
    if "medqa" in dataset_name:
        icl = icl_medqa
    elif "wildchat" in dataset_name:
        icl = icl_wildchat
    else:
        raise

    repeats = 1
    if global_conf.model.lm_model == "llama":
        repeats = 3

    if not global_conf.gpt_only:
        local_gpt_method = "llama-3.1"
        gpt_method = LM(local_gpt_method)

    prompt = PromptWithCache(
        output_path,
        gpt_method,
        repeats=repeats,
        set_alt_job_for_inference=global_conf.set_alt_job_for_inference,
        exp_id=global_conf.transform_dataset.exp_folder,
        bin_id=BIN_ID,
    )
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    while prompt.should_keep_looping():
        # for entry in tqdm(dataset):
        labels = [
            f"{exp_id}-{scores}"
            for scores in [
                "san_fact_on_orig_context",
                "orig_fact_on_san_context",
                "orig_fact_exclude_input_on_san_context",
                "processing_error_rate",
            ]
        ]
        for range_txt, single_range in zip(range_maps, ranges):
            san_fact_on_orig_context = {
                "score": [],
                "missing": [],
            }
            orig_fact_on_san_context = {
                "score": [],
                "missing": [],
            }
            orig_fact_exclude_input_on_san_context = {
                "score": [],
                "missing": [],
            }
            if global_conf.debug:
                queries = []
            in_token = 0
            out_token = 0
            for entry in dataset:
                # top_article_ids = entry["article_idx"]
                # top_article_ids = top_article_ids[single_range]
                # if not top_article_ids:
                #     # If we don't have a valid top article, we skip this entry
                #     continue
                # top_article_idx = mode(list(map(lambda x: x[0], top_article_ids)))
                top_article_idx = get_top_article_idx(entry, single_range)
                if top_article_idx is None:
                    continue
                this_article_id = entry["id"]
                context_orig = entry[context_key_orig]
                # breakpoint()
                facts_orig = entry[facts_key_orig]
                context_sanitized = dataset[top_article_idx][context_key_sanitized]
                facts_sanitized = dataset[top_article_idx][facts_key_sanitized]




                if do_precision:
                    precision_prompts = list(map(
                        lambda x: formulate_entailment_prompt(context_orig, [x], icl),
                        facts_sanitized,
                    ))


                recall_prompts = []
                recall_prompts_expected_len = []
                for batch_idx in range(0, len(facts_orig), BATCH_SIZE):
                    batch_facts = facts_orig[batch_idx:batch_idx + BATCH_SIZE]
                    recall_prompts.append(
                        formulate_entailment_prompt(context_sanitized, batch_facts, icl)
                    )
                    recall_prompts_expected_len.append(len(batch_facts))
                # res_precision_prompt = prompt.prompt(precision_prompt)
                # res_recall_prompt = prompt.prompt(recall_prompt)
                if do_precision:
                    res_precision_prompts = list(map(prompt.prompt, precision_prompts))
                res_recall_prompts = list(map(prompt.prompt, recall_prompts))
                if not res_recall_prompts or res_recall_prompts[0] is None:
                    continue

                for recall_prompt in recall_prompts:
                    in_token += len(enc.encode(recall_prompt))
                for res_recall_prompt in res_recall_prompts:
                    out_token += len(enc.encode(res_recall_prompt[0]))
                # if res_precision_prompt is None:
                #     continue
                # precision_scores = post_process(res_precision_prompt)
                if global_conf.debug:
                    queries.append(
                        {
                            "recall_prompts": recall_prompts,
                            "res_recall_prompts": res_recall_prompts,
                        }
                    )

                if do_precision:
                    precision_scores = map(post_process, res_precision_prompts, precision_prompts)
                    
                    precision_scores = list(map(flatten_individual_scores, precision_scores))                
                    assert len(precision_scores) == len(facts_sanitized)

                recall_scores = list(map(post_process, res_recall_prompts, recall_prompts, recall_prompts_expected_len))
                
                recall_scores = list(map(ensure_scores, recall_scores, recall_prompts_expected_len))                
                recall_scores = [score for scores in recall_scores for score in scores]
                recall_scores = recall_scores[:len(facts_orig)]
                
                # This must be true given the setup, so we can remove the code block below
                assert len(recall_scores) == len(facts_orig)

                r_full = list(range(len(recall_scores)))
                # r_selected = r_full[single_range]
                aux_range = cfc_range if cfc_range else single_range
                r_selected = proper_slice_array_by_range(aux_range, entry, r_full)
                r_complement = list(set(r_full) - set(r_selected))
                recall_exclude_input_scores = [
                    recall_scores[idx] for idx in r_complement
                ]

                if do_precision:
                    if len(precision_scores) > 0:
                        precision_scores_not_missing = list(
                            filter(lambda x: x >= 0, precision_scores)
                        )
                        if precision_scores_not_missing:
                            san_fact_on_orig_context["score"].append(
                                sum(precision_scores_not_missing)
                                / len(precision_scores_not_missing)
                            )
                        san_fact_on_orig_context["missing"].append(
                            1 - len(precision_scores_not_missing) / len(precision_scores)
                        )
                if len(recall_scores) > 0:
                    recall_scores_not_missing = list(
                        filter(lambda x: x >= 0, recall_scores)
                    )

                    if recall_scores_not_missing:
                        orig_fact_on_san_context["score"].append(
                            sum(recall_scores_not_missing)
                            / len(recall_scores_not_missing)
                        )
                    else:
                        continue

                    orig_fact_on_san_context["missing"].append(
                        1 - len(recall_scores_not_missing) / len(recall_scores)
                    )
                if len(recall_exclude_input_scores) > 0:
                    recall_exclude_input_scores_not_missing = list(
                        filter(lambda x: x >= 0, recall_exclude_input_scores)
                    )
                    if recall_exclude_input_scores_not_missing:
                        orig_fact_exclude_input_on_san_context["score"].append(
                            sum(recall_exclude_input_scores_not_missing)
                            / len(recall_exclude_input_scores_not_missing)
                        )
                    orig_fact_exclude_input_on_san_context["missing"].append(
                        1
                        - len(recall_exclude_input_scores_not_missing)
                        / len(recall_exclude_input_scores)
                    )

            if prompt.resolved:
                if global_conf.debug:
                    breakpoint()
                print("in token: ", in_token)
                print("out token: ", out_token)
                print("in_token_price: ", in_token * 5 / 1000000)
                print("out_token_price: ", out_token * 15 / 1000000)
                print("sum_of_price: ", (in_token * 5 + out_token * 15) / 1000000)
                print(f'error rate: {errors /calls}')
                data = [
                    str(sum(scores[cat]) / len(scores[cat])) if scores[cat] else "N/A"
                    for cat in ["score"]
                    for scores in [
                        san_fact_on_orig_context,
                        orig_fact_on_san_context,
                        orig_fact_exclude_input_on_san_context,
                    ]
                ]
                data.append(str(errors / calls))
                result[range_txt] = dict(zip(labels, data))

    # exit()
    if result:
        print_and_write_results_json(result)
        print_and_write_results(result)
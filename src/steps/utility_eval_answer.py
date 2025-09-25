from src.utils import *
from functools import partial
import re
from rouge_score import rouge_scorer
import numpy as np
from src.gpt import LM


def post_process_specialty_answers(
    responses, labels=None, subspecialties=None, **kwargs
):
    llama_pred = responses

    specialty = None
    for label in labels:
        if label in llama_pred:
            specialty = label
            break

    else:
        # Search for the proper subspecialty
        for i, subspecialty_list in enumerate(subspecialties):
            for subspecialty in subspecialty_list:
                if subspecialty in llama_pred:
                    specialty = labels[i]
                    break
            else:
                continue
            break
        else:
            specialty = "Others"

    return specialty


def post_process_medical_quality_answers(response, **kwargs):
    pattern = re.compile(r"rating.*?(\d)", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(response)
    if match:
        return match.group(1)
    return None


scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

categories = [
    "summarization",
    "model jailbreaking",
    "generating prompts for AI models",
    "story and script generation",
    "song and poem generation",
    "generating character descriptions",
    "code generation",
    "code editing and debugging",
    "generating communications",
    "generating non-fictional documents",
    "editing existing text",
    "comparison, ranking, and recommendation",
    "brainstorming and generating ideas",
    "information retrieval",
    "solving logic, math, and word problems",
    "explanation, how-to, practical advice",
    "personal advice about mental health, relationships, etc.",
    "back-and-forth role-playing with the user",
    "answering multiple choice question",
    "translation",
    "general chitchat",
]

# categories = [
#     "code generation",
#     "generating non-fictional documents",
#     "explanation, how-to, practical advice",
#     "editing existing text",
#     "information retrieval",
#     "generating prompts for ai models",
#     "personal advice about mental health, relationships, etc.",
#     "translation",
#     "back-and-forth role-playing with the user",
#     "code editing and debugging",
#     "story and script generation",
#     "summarization",
#     "generating communications",
#     "song and poem generation",
#     "brainstorming and generating ideas",
#     "model jailbreaking",
#     "generating character descriptions",
#     "solving logic, math, and word problems",
#     "grading an essay",
#     "general chitchat",
#     "comparison, ranking, and recommendation",
#     "answering multiple choice question",
#     "explanation",
#     "generating multiple choice questions",
#     "proofreading",
#     "debate generation",
#     "summariation",
# ]


def post_process_wildchat_answers(i):
    i = i.lower()
    i = i.split("**task")
    if len(i) > 1:
        i = i[1]
    else:
        i = i[0]
    i = i.split('"task')
    if len(i) > 1:
        i = i[1]
    else:
        i = i[0]
    task = i.split("confidence")[0]
    # print(task)
    recalls = []
    for category in categories:
        # Remove the samples as they are sometimes not generated
        recalls.append(scorer.score(category.split("(")[0], task)["rouge1"].recall)
    selected_task = categories[np.argmax(recalls)]
    # if "model jailbreaking" in selected_task:
    #     breakpoint()
    return selected_task


def formulate_sample(entry, test=False, key="context"):
    options = entry["options"]
    if isinstance(options, list):
        formatted_options = {option["key"]: option["value"] for option in options}
        options = formatted_options

    question = entry["question"]
    if callable(key):
        question = key(entry) + " " + question

    elif key in entry:
        question = entry[key] + " " + question

    return f"""Question:
{question}
Options:
A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}

Answer: {f"[INST]{entry['answer_idx']}[INST]" if not test else ""}"""


# In the future it should evolve to include more metrics. Maybe re-write it into hydra-zen?
# Validation should get written either as a type check or a registry
ANSWER_TYPES = ["qa", "specialty", "medical_quality", "wildchat_label_distribution", 'quality']
# For the QUE methods, we only want to evaluate the answer type that the QUE method decided
QUE_DECIDED_ANSWER_TYPES = ["qa", "specialty"]


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
    exp_name_config,
    dataset_name,
    **kwargs,
):
    args = step_conf.method.split("_")[1:]
    answer_type = method_name_config["answer"]
    empty_context = method_name_config.get("empty", False)
    # For dpft methods we need to match the way it's prompted in the training
    # step as measure the performance of the other does not make sense.
    # exp_name_config contains ALL previous step config
    if "cued" in exp_name_config:
        cued_type = exp_name_config["cued"]
        if answer_type in QUE_DECIDED_ANSWER_TYPES and answer_type != cued_type:
            return STEP_FLAGS.SKIP_CHILD

    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"Answer type {answer_type} not supported")

    exp_name = step_conf.exp_name
    if "medqa" in dataset_name:
        if answer_type == "quality":
            if empty_context:
                raise "not Reachable"
            local_gpt_method = LM("gpt-4o-2024-08-06")

            def formulate_sample_pmc(record):
                context = access_func(record)
                # labels = record["choices"]
                # icl_samples = """
                # What is the diagnosis? Select one answer among Bronchitis, Tuberculosis, Chronic rhinosinusitis, SLE.[INST]Bronchitis[INST]
                # What is the diagnosis? Select one answer among Ebola, Possible NSTEMI / STEMI, Myasthenia gravis, Boerhaave.[INST]Myasthenia gravis[INST]
                # What is the diagnosis? Select one answer among Acute rhinosinusitis, Pericarditis, Stable angina, Cluster headache.[INST]Stable angina[INST]
                # """
                util_eval_blurb = f"""You are provided with a concise medical case summary. Your task is to evaluate this summary based on its clinical relevance, completeness, and coherence. Use the following 5-point scale:
1 (Very Poor): The summary is severely lacking in critical information, contains major contradictions, or is so incomplete that it fails to provide a meaningful picture of the patient's condition.
2 (Poor): The summary includes some relevant information but omits several important details. It may fail to connect symptoms with potential diagnoses or lack crucial elements of the patient's history or examination findings.
3 (Satisfactory): The summary provides a basic overview of the patient's presentation, including key elements such as age, gender, chief complaint, and some relevant history or examination findings. However, it may lack depth or miss some important details.
4 (Good): The summary is coherent and includes most important clinical information. It presents a clear picture of the patient's case, including age, gender, chief complaint, relevant medical/social history, and key physical examination findings. Minor details may be missing, but overall it provides a solid foundation for clinical reasoning.
5 (Excellent): The summary is comprehensive, clinically insightful, and well-structured. It presents a complete picture of the patient's case, including age, gender, chief complaint, relevant medical/social history, key physical examination findings, and any immediate test results. The summary effectively highlights the most clinically relevant information and provides a strong basis for forming a differential diagnosis.
Evaluation Instructions:

Read the case summary carefully.
Assess the summary based on the above criteria, focusing on its clinical relevance, completeness, and coherence.
Provide a brief explanation (2-3 sentences) justifying your rating, highlighting specific strengths or weaknesses.
Assign a rating from 1 to 5.

Your response should follow this format:
Explanation: [Your 2-3 sentence justification]
Rating: [Your rating (1-5)]
Remember, you are evaluating the quality and completeness of the case summary itself, not making a diagnosis or judging the medical decisions. Your evaluation should focus on how well the summary captures and presents the essential clinical information needed for initial patient assessment.

Medical Profile: {context}
"""
                return util_eval_blurb

            prompt = Prompt(
                "{}",
                post_process_func=post_process_medical_quality_answers,
            )

            dataset = prompt.apply(
                dataset, local_gpt_method, formulate_sample_pmc, output_dataset_key
            )
        elif answer_type == "specialty":
            if empty_context:
                # supports empty context
                pass

            with open(
                "/mmfs1/gscratch/sewoong/rx31/projects/dynamed/src/info_seek/specialties.json"
            ) as f:
                specialties_dict = json.load(f)

            labels = list(
                map(lambda x: x["name"], specialties_dict["medical_specialties"])
            ) + ["Others"]
            subspecialties = list(
                map(
                    lambda x: x["subspecialties"],
                    specialties_dict["medical_specialties"],
                )
            )

            # last_samples = [dataset[i] for i in range(-3, 0)]

            icl_examples = [
                {
                    "context": "After receiving a positive newborn screening result, a 2-week-old male infant is brought to the pediatrician for a diagnostic sweat test. The results demonstrated chloride levels of 65 mmol/L (nl < 29 mmol/L). Subsequent DNA sequencing revealed a 3 base pair deletion in a transmembrane cAMP-activated ion channel known to result in protein instability and early degradation. The physician discusses with the parents that the infant will develop respiratory infections due to improper mucus clearance and reviews various mucolytic agents, such as one that cleaves disulfide bonds between mucus glycoproteins thereby loosening the mucus plug.",
                    "answer": "Pediatrics",
                },
                {
                    "context": "A 25-year-old man comes to the office because of pain in his left shoulder. He says that this pain started 3 years ago and has progressively worsened. He denies joint trauma, fever, dysuria, or morning stiffness. He says that his urine turns black after it is exposed to air and has done so since childhood. He has one sexual partner and they regularly use condoms. His pulse is 72/min, respiratory rate is 18/min, temperature is 37.2°C (99.0°F), and blood pressure is 135/80 mm Hg. Physical examination shows bilateral scleral darkening and point tenderness upon palpation of his right elbow, left knee, and shoulder. Leukocyte count is 6,000/mm3.",
                    "answer": "Internal Medicine",
                },
                {
                    "context": "A 26-year-old primigravid woman comes to the emergency department because of a 10-hour history of vaginal bleeding and lower abdominal pain. She also had nausea and fatigue for the past 4 weeks. Her last menstrual period was 9 weeks ago. There is no history of medical illness. Vital signs are within normal limits. Pelvic examination shows a uterus consistent in size with a 9-week gestation. A urine pregnancy test is positive. β-HCG level is 108,000 mIU/mL (N < 5 mIU/mL). Transvaginal ultrasonography shows unclear, amorphous fetal parts and a large placenta with multiple cystic spaces.",
                    "answer": "Obstetrics and Gynecology",
                },
            ]

            def construct_prompt_same(train_examples, test_example):
                # prompt = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
                prompt = f"Classify the medical records into the categories of {', '.join(labels[:-1])}, and {labels[-1]}.\n\n"
                for train_example in train_examples:
                    if empty_context:
                        prompt += "Article: empty\n"
                    else:
                        prompt += "Article: " + train_example["context"] + "\n"
                    prompt += "Answer: [INST]" + train_example["answer"] + "[INST]"
                if empty_context:
                    prompt += "Article: empty\n"
                else:
                    prompt += "Article: " + access_func(test_example) + "\n"
                prompt += "Answer:"
                return prompt

            formulate_prompt = lambda entry: construct_prompt_same(icl_examples, entry)
            prompt = Prompt(
                "{}",
                post_process_func=partial(
                    post_process_specialty_answers,
                    labels=labels,
                    subspecialties=subspecialties,
                ),
            )
            dataset = prompt.apply(
                dataset, gpt_method, formulate_prompt, output_dataset_key
            )

        elif answer_type == "qa":
            if empty_context:
                # supports empty context
                pass
            arg = ""
            if len(args) == 1:
                arg = args[0]

            # breakpoint()
            last_samples = [dataset[i] for i in range(-3, 0)]
            icl_samples = "\n\n".join(
                [formulate_sample(entry) for entry in last_samples]
            )
            user_prompt_instr = "Solve the following medical multiple choice question and answer correctly:\n\n"

            # field = "context"
            # access_func = (
            #     lambda entry: f"""{user_prompt_instr}{icl_samples}\n\n{formulate_sample(entry, test=True, key=field)}"""
            # )
            # dataset = dataset.map(
            #     wrap_dataset_map_func(
            #         do_inference,
            #         access_func,
            #         f"{output_dataset_key}_{field}_responses",
            #         gpt_method,
            #         "{}",
            #     )
            # )

            # field = "processed_sanitized_context"
            # if method_name_config['empty']:
            if empty_context:
                field = "should_not_exist"
            else:
                field = access_func
            access_func = (
                lambda entry: f"""{user_prompt_instr}{icl_samples}\n\n{formulate_sample(entry, test=True, key=field)}"""
            )
            prompt = Prompt("{}")
            dataset = prompt.apply(
                dataset,
                gpt_method,
                access_func,
                output_dataset_key,
                is_mcq=True,
                mcq_choices=["A", "B", "C", "D"],
            )
        else:
            return STEP_FLAGS.SKIP_CHILD

    elif "ddxp" in dataset_name:
        if answer_type == "qa":
            if empty_context:

                def formulate_sample_ddxp_empty(record):
                    labels = record["choices"]

                    icl_samples = """
What is the diagnosis? Select one answer among Bronchitis, Tuberculosis, Chronic rhinosinusitis, SLE.[INST]Bronchitis[INST]
What is the diagnosis? Select one answer among Ebola, Possible NSTEMI / STEMI, Myasthenia gravis, Boerhaave.[INST]Myasthenia gravis[INST]
What is the diagnosis? Select one answer among Acute rhinosinusitis, Pericarditis, Stable angina, Cluster headache.[INST]Stable angina[INST]
"""

                    util_eval_blurb = f"""
The following is a patient's information and dialog with the doctor.

{icl_samples} 

What is the diagnosis? Select one answer among {', '.join(labels)}."""
                    return util_eval_blurb

                prompt = Prompt("{}")

                dataset = prompt.apply(
                    dataset, gpt_method, formulate_sample_ddxp_empty, output_dataset_key
                )
            else:

                def formulate_sample_ddxp(record):
                    context = access_func(record)

                    labels = record["choices"]

                    icl_samples = """
Age: 47. Sex: M. Do you have pain somewhere, related to your reason for consulting? Yes. Do you have pain somewhere, related to your reason for consulting? Yes. Characterize your pain: burning. Do you feel pain somewhere? lower chest. Do you feel pain somewhere? side of the chest(L). Do you feel pain somewhere? upper chest. Do you feel pain somewhere? posterior chest wall(R). How intense is the pain? 5. Does the pain radiate to another location? nowhere. How precisely is the pain located? 6. How fast did the pain appear? 5. Are you experiencing shortness of breath or difficulty breathing in a significant way? Yes. Do you have a cough that produces colored or more abundant sputum than usual? Yes. Do you smoke cigarettes? Yes. Do you have a chronic obstructive pulmonary disease (COPD)? Yes. Do you work in agriculture? Yes. Do you work in construction? Yes. Do you have a cough? Yes. Have you traveled out of the country in the last 4 weeks? N. Are you immunosuppressed? Yes. What is the diagnosis? Select one answer among Bronchitis, Tuberculosis, Chronic rhinosinusitis, SLE.[INST]Bronchitis[INST]
Age: 28. Sex: M. Did you previously, or do you currently, have any weakness/paralysis in one or more of your limbs or in your face? Yes. Are there any members of your family who have been diagnosed myasthenia gravis? Yes. Do you have pain or weakness in your jaw? Yes. Do you have the perception of seeing two images of a single object seen overlapping or adjacent to each other (double vision)? Yes. Do you have difficulty swallowing, or have a feeling of discomfort/blockage when swallowing? Yes. Do your symptoms of muscle weakness increase with fatigue and/or stress? Yes. Do you have a hard time opening/raising one or both eyelids? Yes. Did you previously, or do you currently, have any weakness/paralysis in one or more of your limbs or in your face? Yes. Do you currently, or did you ever, have difficulty swallowing, or have a feeling of discomfort/blockage when swallowing? Yes. Do you feel weakness in both arms and/or both legs? Yes. Did you previously, or do you currently, have any weakness/paralysis in one or more of your limbs or in your face? Yes. Do you currently, or did you ever, have numbness, loss of sensitivity or tingling anywhere on your body? Yes. Have you traveled out of the country in the last 4 weeks? N. What is the diagnosis? Select one answer among Ebola, Possible NSTEMI / STEMI, Myasthenia gravis, Boerhaave.[INST]Myasthenia gravis[INST]
Age: 55. Sex: F. Do you have symptoms that are increased with physical exertion but alleviated with rest? Yes. Do you have pain somewhere, related to your reason for consulting? Yes. Characterize your pain: heavy. Characterize your pain: exhausting. Do you feel pain somewhere? biceps(R). Do you feel pain somewhere? thyroid cartilage. Do you feel pain somewhere? side of the chest(L). Do you feel pain somewhere? upper chest. Do you feel pain somewhere? epigastric. How intense is the pain? 5. Does the pain radiate to another location? biceps(R). Does the pain radiate to another location? side of the chest(L). Does the pain radiate to another location? upper chest. How precisely is the pain located? 6. How fast did the pain appear? 5. Do you have a known issue with one of your heart valves? Yes. Are you experiencing shortness of breath or difficulty breathing in a significant way? Yes. Do you feel your heart is beating fast (racing), irregularly (missing a beat) or do you feel palpitations? Yes. Do you feel your heart is beating very irregularly or in a disorganized pattern? Yes. Have you traveled out of the country in the last 4 weeks? N. What is the diagnosis? Select one answer among Acute rhinosinusitis, Pericarditis, Stable angina, Cluster headache.[INST]Stable angina[INST]
"""

                    util_eval_blurb = f"""
The following is a patient's information and dialog with the doctor.

{icl_samples} 

{context} What is the diagnosis? Select one answer among {', '.join(labels)}."""
                    return util_eval_blurb

                prompt = Prompt("{}")

                dataset = prompt.apply(
                    dataset, gpt_method, formulate_sample_ddxp, output_dataset_key
                )
        else:
            return STEP_FLAGS.SKIP_CHILD

    elif "pmc" in dataset_name:
        if answer_type == "medical_quality":
            if empty_context:
                raise "not Reachable"

            def formulate_sample_pmc(record):
                context = access_func(record)
                # labels = record["choices"]
                # icl_samples = """
                # What is the diagnosis? Select one answer among Bronchitis, Tuberculosis, Chronic rhinosinusitis, SLE.[INST]Bronchitis[INST]
                # What is the diagnosis? Select one answer among Ebola, Possible NSTEMI / STEMI, Myasthenia gravis, Boerhaave.[INST]Myasthenia gravis[INST]
                # What is the diagnosis? Select one answer among Acute rhinosinusitis, Pericarditis, Stable angina, Cluster headache.[INST]Stable angina[INST]
                # """
                util_eval_blurb = f"""Case Summary: {context}
You are provided with a concise medical case summary. Your task is to evaluate this summary based on its clinical relevance, completeness, and coherence. Use the following criteria:
Unacceptable (1 point): The summary omits critical clinical information, contains significant contradictions, or fails to provide a clear picture of the patient's presentation and key findings.
Poor (2 points): The summary includes some relevant information but lacks important details. It may fail to connect symptoms with potential diagnoses or omit crucial elements of the patient's history or examination findings.
Satisfactory (3 points): The summary provides a coherent overview of the patient's presentation, including most key elements such as age, gender, chief complaint, relevant history, and significant examination findings. It may lack some depth or miss minor details.
Excellent (4 points): The summary is comprehensive and clinically insightful. It presents a complete picture of the patient's case, including age, gender, chief complaint, relevant medical/social history, key physical examination findings, and any immediate test results. The summary effectively highlights the most clinically relevant information for forming a differential diagnosis.
Evaluation Instructions:

Read the case summary carefully.
Assess the summary based on the above criteria, focusing on its clinical relevance, completeness, and coherence.
Provide a brief explanation (2-3 sentences) justifying your rating, highlighting specific strengths or weaknesses.
Assign a rating from 1 to 4.

Your response should follow this format:
Explanation: [Your 2-3 sentence justification]
Rating: [Your rating (1-4)]
Remember, you are evaluating the quality and completeness of the case summary itself, not making a diagnosis or judging the medical decisions. Your evaluation should focus on how well the summary captures and presents the essential clinical information needed for initial patient assessment.

"""
                return util_eval_blurb

            prompt = Prompt(
                "{}",
                post_process_func=post_process_medical_quality_answers,
            )

            dataset = prompt.apply(
                dataset, gpt_method, formulate_sample_pmc, output_dataset_key
            )
        else:
            return STEP_FLAGS.SKIP_CHILD

    elif "wildchat" in dataset_name:
        if answer_type == "quality":

            if empty_context:
                raise "not Reachable"
            local_gpt_method = LM("gpt-4o-2024-08-06")

            def formulate_sample_pmc(record):
                context = access_func(record)
                # labels = record["choices"]
                # icl_samples = """
                # What is the diagnosis? Select one answer among Bronchitis, Tuberculosis, Chronic rhinosinusitis, SLE.[INST]Bronchitis[INST]
                # What is the diagnosis? Select one answer among Ebola, Possible NSTEMI / STEMI, Myasthenia gravis, Boerhaave.[INST]Myasthenia gravis[INST]
                # What is the diagnosis? Select one answer among Acute rhinosinusitis, Pericarditis, Stable angina, Cluster headache.[INST]Stable angina[INST]
                # """
                util_eval_blurb = f"""Your task is to evaluate the quality of synthetic conversations, including both the user input and the AI response. Each entry represents a complete exchange. Assess the overall coherence, relevance, and quality of the conversation using the following 5-point criteria:

Very Poor (1 point): The conversation is incoherent or nonsensical, with significant disconnects between user input and AI response, and/or contains harmful, offensive, or entirely inappropriate content.
Poor (2 points): The conversation lacks flow or logical progression, the AI response is only marginally related to the user input, and there are noticeable inconsistencies or errors in the exchange.
Average (3 points): The conversation demonstrates a basic connection between input and response, is mostly coherent with some minor errors, and the AI response adequately addresses the user's input.
Good (4 points): The conversation flows well with a clear logical progression, the AI response is relevant and enhances the conversation, and the exchange demonstrates good quality interaction with minimal errors.
Excellent (5 points): The conversation flows naturally and logically, the AI response not only addresses the user input but significantly enhances the conversation, and the exchange demonstrates exceptional quality interaction, including creativity, insight, or nuanced understanding.

Evaluation Instructions:

Carefully read the entire conversation exchange.
Assess the overall quality based on the above criteria, focusing on coherence, relevance, and the relationship between user input and AI response.
Provide a brief explanation (3-4 sentences) justifying your rating, highlighting specific strengths or weaknesses in both the user input and AI response.
Assign a rating from 1 to 5.

Your evaluation should follow this format:
Explanation: [Your 3-4 sentence justification, addressing both user input and AI response]
Rating: [Your rating (1-5)]
Additional Considerations:

Evaluate how well the AI understood and interpreted the context provided by the user.
Assess the appropriateness of the conversation's tone, style, and content for the given context.
Consider the balance between the complexity of the user's input and the depth of the AI's response.
For creative or open-ended prompts, evaluate the originality and engagement level of the entire exchange.
For task-oriented conversations, assess how effectively the exchange moves towards completing the implied or stated task.

Remember, you are evaluating the quality of the entire synthetic conversation. Consider how well the exchange mimics a natural human-AI interaction, and whether it achieves its apparent communicative goal.

Conversation: {context}"""
                return util_eval_blurb

            prompt = Prompt(
                "{}",
                post_process_func=post_process_medical_quality_answers,
            )

            dataset = prompt.apply(
                dataset, local_gpt_method, formulate_sample_pmc, output_dataset_key
            )
        elif answer_type == "wildchat_label_distribution":
            # if empty_context:
            #     raise "not Reachable"

            def formulate_sample_pmc(record):
                context = access_func(record)[:5000]
                if empty_context:
                    context = ''
                # labels = record["choices"]
                # icl_samples = """
                # What is the diagnosis? Select one answer among Bronchitis, Tuberculosis, Chronic rhinosinusitis, SLE.[INST]Bronchitis[INST]
                # What is the diagnosis? Select one answer among Ebola, Possible NSTEMI / STEMI, Myasthenia gravis, Boerhaave.[INST]Myasthenia gravis[INST]
                # What is the diagnosis? Select one answer among Acute rhinosinusitis, Pericarditis, Stable angina, Cluster headache.[INST]Stable angina[INST]
                # """
                # util_eval_blurb = f"""Read the following conversation between a user and an AI chatbot. Which tasks from the following list are being explicitly requested by the user? For each task, list the task, your confidence, and your reasoning and evidence.
                util_eval_blurb = f"""Read the following conversation between a user and an AI chatbot. Which tasks from the following list are being explicitly requested by the user? Return only the most likely task name.

Tasks: 
- summarization
- model jailbreaking (e.g. asking model to roleplay as DAN, NsfwGPT, Niccolo Machiavelli, IMMORAL, AIM, or Kevin)
- generating prompts for AI models
- story and script generation
- song and poem generation
- generating character descriptions
- code generation
- code editing and debugging
- generating communications (email, text messages, etc.)
- generating non-fictional documents (resumes, essays, etc.)
- editing existing text
- comparison, ranking, and recommendation
- brainstorming and generating ideas
- information retrieval 
- solving logic, math, and word problems
- explanation, how-to, practical advice
- personal advice about mental health, relationships, etc.
- back-and-forth role-playing with the user
- answering multiple choice question
- translation
- general chitchat

Conversation:
{context}

Answer:"""
                return util_eval_blurb

            prompt = Prompt(
                "{}",
                post_process_func=post_process_wildchat_answers,
            )

            local_gpt_method = LM("gpt-4o-2024-08-06")
            # local_gpt_method = gpt_method
            dataset = prompt.apply(
                dataset,
                local_gpt_method,
                formulate_sample_pmc,
                output_dataset_key,
                # processing_args={"num_proc": 6, "batch_size": 10, "batched": True},
            )
        else:
            return STEP_FLAGS.SKIP_CHILD

    else:
        raise

    return dataset

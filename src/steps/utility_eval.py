from src.utils import *
from rouge_score import rouge_scorer
import numpy as np
from src.load_dataset import load_dataset_from_config
from functools import partial
from collections import defaultdict



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
    exp_name = step_conf.exp_name
    if 'medqa' in dataset_name:
        if "quality" in exp_name:
            correct_sanitized_context = 0
            issue = 0
            for entry in dataset:
                ans_processed_sanitized_context = access_func(entry)
                if ans_processed_sanitized_context is None:
                    # If the score cannot get computed, we assume it is 1 (worst)
                    # ans_processed_sanitized_context = "1"
                    issue += 1
                    continue
                correct_sanitized_context += (
                    int(ans_processed_sanitized_context)
                )
                # print_and_write_results(
                #     f"Sanitized Context: {ans_processed_sanitized_context}, True Answer: {true_ans}"
                # )
            print_and_write_results(f"{issue} / {len(dataset)}")
            # print_and_write_results(f"{correct_sanitized_context/len(dataset)}")
            print_and_write_results_json(
                {
                    "utility_quality": correct_sanitized_context / (len(dataset) - issue),
                }
            )
            
        elif "specialty" in exp_name:
            # if specialty is None:
            #     specialty = "Others"

            correct_sanitized_context = 0
            for entry in dataset:
                ans_processed_sanitized_context = access_func(entry).lower()
                true_ans = entry["patient"]["specialty"].lower()
                correct_sanitized_context += (
                    1 if true_ans in ans_processed_sanitized_context else 0
                )
                print_and_write_results(
                    f"Sanitized Context: {ans_processed_sanitized_context}, True Answer: {true_ans}"
                )

            print_and_write_results(f"{correct_sanitized_context/len(dataset)}")
            print_and_write_results_json(
                {
                    "utility_specialty": correct_sanitized_context / len(dataset),
                }
            )
        else:
            correct_context = 0
            correct_sanitized_context = 0
            # answer_changed = 0
            # cached_truth_dataset_cfg = conf.dataset_in.copy()
            # cached_truth_dataset_cfg.path = (
            #     "data/answer-med_qa_factorized-llama-1000/dataset"
            # )
            # cached_truth_dataset_cfg.key = "answer-med_qa_factorized-llama-1000"
            # cached_truth_dataset, cached_truth_dataset_access_func = (
            #     load_dataset_from_config(cached_truth_dataset_cfg)
            # )

            # for entry, entry_truth in zip(dataset, cached_truth_dataset):
            for entry in dataset:
                # assert entry["context"] == entry_truth["context"]
                # Works better with chatgpt lol. was due to agent switching probably
                # ans_context = cached_truth_dataset_access_func(entry_truth).split("\n")[
                #     -1
                # ][0]
                # ans_processed_sanitized_context = access_func(entry).split("\n")[-1][0]
                # ans_context = cached_truth_dataset_access_func(entry_truth)[0]


                if access_func(entry) is None:
                    continue
                ans_processed_sanitized_context = access_func(entry)[0]
                # ans_processed_sanitized_context = entry['answer-empty-med_qa_factorized-llama-1000'][0]
                true_ans = entry["answer_idx"]
                # correct_context += 1 if ans_context == true_ans else 0
                correct_sanitized_context += (
                    1 if ans_processed_sanitized_context == true_ans else 0
                )
                # answer_changed += (
                #     1 if ans_context != ans_processed_sanitized_context else 0
                # )
                print_and_write_results(
                    f"Sanitized Context: {ans_processed_sanitized_context}, True Answer: {true_ans}"
                )
            print_and_write_results(
                f"Correct context: {correct_context}/{len(dataset)}"
            )
            print_and_write_results(
                f"Correct sanitized context: {correct_sanitized_context}/{len(dataset)}"
            )
            # print_and_write_results(f"Answer changed: {answer_changed}/{len(dataset)}")
            # print(f'{correct_context/len(dataset)}\t{correct_sanitized_context/len(dataset)}\t{answer_changed/len(dataset)}')
            print_and_write_results(
                # f"{correct_sanitized_context/len(dataset)}\t{answer_changed/len(dataset)}"
                # Should not keep track of the answer changed metric: not useful 
                f"{correct_sanitized_context/len(dataset)}"
            )
            print_and_write_results_json(
                {
                    "utility_qa": correct_sanitized_context / len(dataset),
                }
            )

    elif 'ddxp' in dataset_name:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        # if specialty is None:
        #     specialty = "Others"

        correct_sanitized_context = 0
        for entry in dataset:
            labels = entry['choices']
            ans = access_func(entry)
            normalized_ans = labels[np.argmax(list(map(lambda x: max(x.precision, x.recall), map(lambda x: scorer.score(ans, x)['rouge1'], labels))))]
            true_ans = entry['answer']
            correct_sanitized_context += normalized_ans == true_ans
            print_and_write_results(
                f"Sanitized Context: {normalized_ans}, True Answer: {true_ans}"
            )

        print_and_write_results(f"{correct_sanitized_context/len(dataset)}")
        print_and_write_results_json(
            {
                "utility_qa": correct_sanitized_context / len(dataset),
            }
        )
        
    elif 'pmc' in dataset_name:
        correct_sanitized_context = 0
        issue = 0
        for entry in dataset:
            ans_processed_sanitized_context = access_func(entry)
            if ans_processed_sanitized_context is None:
                ans_processed_sanitized_context = "1"
                issue += 1
            correct_sanitized_context += (
                int(ans_processed_sanitized_context)
            )
            # print_and_write_results(
            #     f"Sanitized Context: {ans_processed_sanitized_context}, True Answer: {true_ans}"
            # )
        print_and_write_results(f"{issue} / {len(dataset)}")
        # print_and_write_results(f"{correct_sanitized_context/len(dataset)}")
        print_and_write_results_json(
            {
                "utility_medical_quality": correct_sanitized_context / len(dataset),
            }
        )
    elif "wildchat" in dataset_name:
        if "quality" in exp_name:
            correct_sanitized_context = 0
            issue = 0
            for entry in dataset:
                ans_processed_sanitized_context = access_func(entry)
                if ans_processed_sanitized_context is None:
                    # If the score cannot get computed, we assume it is 1 (worst)

                    issue += 1
                    continue
                correct_sanitized_context += (
                    int(ans_processed_sanitized_context)
                )
                # print_and_write_results(
                #     f"Sanitized Context: {ans_processed_sanitized_context}, True Answer: {true_ans}"
                # )
            print_and_write_results(f"{issue} / {len(dataset)}")
            # print_and_write_results(f"{correct_sanitized_context/len(dataset)}")
            print_and_write_results_json(
                {
                    "utility_quality": correct_sanitized_context / (len(dataset) - issue),
                }
            )

        # elif "temp" in exp_name:
        # else:

        #     from scipy.stats import chi2_contingency

            
        #     counts = defaultdict(int)
        #     # counts_orig_1 = defaultdict(int)
        #     #
        #     # return
        #     correct_sanitized_context = 0
        #     issue = 0
        #     for entry in dataset:
        #         ans_processed_sanitized_context = access_func(entry)
        #         if ans_processed_sanitized_context is None:
        #             ans_processed_sanitized_context = "1"
        #             issue += 1
        #         # breakpoint()
        #         counts[ans_processed_sanitized_context] += 1
        #         # counts[entry["predicted_category"].split('(')[0].strip()] += 1

        #     if 'utility_eval-answer_wildchat_label_distribution-sanitize_and_paraphrase-wildchat_all_unprocessed-llama-1000' in output_path.name:
        #         breakpoint()
        #     with open('playground/label_distributions/' + output_path.name + '.tmp', 'wb') as f:
        #         pickle.dump(counts, f)

        else:
            from scipy.stats import chi2_contingency

            
            counts = defaultdict(int)
            # counts_orig_1 = defaultdict(int)
            #
            # return
            correct_sanitized_context = 0
            issue = 0
            for entry in dataset:
                ans_processed_sanitized_context = access_func(entry)
                if ans_processed_sanitized_context is None:
                    ans_processed_sanitized_context = "1"
                    issue += 1
                # breakpoint()
                counts[ans_processed_sanitized_context] += 1
                # counts[entry["predicted_category"].split('(')[0].strip()] += 1

                # correct_sanitized_context += (
                #     int(ans_processed_sanitized_context == entry["predicted_category"])
                # )
                # # if int(ans_processed_sanitized_context == entry["predicted_category"]) == 0:
                # #     breakpoint()
                # print(int(ans_processed_sanitized_context == entry["predicted_category"]), ans_processed_sanitized_context, '||', entry["predicted_category"])
                # # print_and_write_results(
                # #     f"Sanitized Context: {ans_processed_sanitized_context}, True Answer: {true_ans}"
                # # )

            # for entry in dataset:
            #     responses = access_func(entry)
            #     yes_no_list = re.findall(
            #         r"^\d+.*\b(Yes|No)\b", responses, re.MULTILINE | re.IGNORECASE
            #     )
            #     if len(yes_no_list) != len(symptom_list):
            #         print(responses)

            #     yes_no_list = map(lambda x: 1 if x.lower() == "yes" else 0, yes_no_list)
            #     count_dict = dict(zip(symptom_list, yes_no_list))
            #     for i in count_dict:
            #         counts[i] += count_dict[i]

            # # context_key = regex_find(r"[^-]*sanitize", dataset.column_names)
            # context_key = get_access_key(dataset, "sanitized_document", global_conf)
            # if context_key is None:
            #     context_key = "context"
            # counts_em = defaultdict(int)
            # for symptom in symptom_list:
            #     counts_em[symptom] = 0
            # for entry in dataset:
            #     responses = entry[context_key].lower()
            #     for symptom in symptom_list:
            #         if symptom.lower() in responses:
            #             counts_em[symptom] += 1

            # # Pickle counts
            # output_path.mkdir(parents=True, exist_ok=True)
            # with open(output_path / "counts.pickle", "wb") as f:
            #     pickle.dump([counts, counts_em], f)

            pickle_path = 'data/wildchat/categories.pkl'

            with open(
                pickle_path,
                "rb",
            ) as f:
                counts_orig = pickle.load(f)
            # Compute chi-square for counts
            # breakpoint()
            # assert counts.keys() == counts_orig.keys()
            for key in counts.keys():
                assert key in counts_orig.keys()
            for key in counts_orig.keys():
                if key not in counts:
                    counts[key] = 0

            # normalize the count values
            tot = sum(counts_orig.values())
            for i in counts_orig:
                counts_orig[i] = counts_orig[i] / tot * 1000

            # observed_counts = [list(counts.values()), list(counts_orig.values())]
            data_pairs = []
            for key in counts_orig.keys():
                data_pairs.append((counts[key], counts_orig[key]))
            observed_counts = list(zip(*data_pairs))
            # breakpoint()

            # breakpoint()
            # observed_counts = list(
            #     zip(*list(filter(lambda x: x[1], zip(*observed_counts))))
            # )
            chi2, p_value, _, _ = chi2_contingency(observed_counts)

            print_and_write_results(f"Chi-square value: {chi2}")
            print_and_write_results(f"P-value: {p_value}")
            blurb_1 = f"{chi2}\t{p_value}"
            # breakpoint()

            # print_and_write_results(f"Chi-square value for em measure: {chi2}")
            # print_and_write_results(f"P-value for em measure: {p_value}")
            # print_and_write_results(f"{blurb_1}\t{chi2}\t{p_value}")

            # breakpoint()
            # print_and_write_results(f"{issue} / {len(dataset)}")
            print_and_write_results(blurb_1)
            # print_and_write_results(f"{correct_sanitized_context/len(dataset)}")
            print_and_write_results_json(
                {
                    # "wildchat_temp": correct_sanitized_context / len(dataset),
                    "chi2": chi2,
                    "p_value": p_value,
                }
            )

    return

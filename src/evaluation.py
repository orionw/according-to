
from kilt.eval_downstream import _metric_max_over_ground_truths, _exact_match_score, _f1_score, _rougel_score
from sacrebleu import corpus_bleu
import json
import sys
sys.setrecursionlimit(9999 * 9999)

from check_datasketch import init_sketch, check

# ds = init_sketch()


def do_kilt_eval(outputs, refs, dataset_name: str, output_format: str):
    # adapted from https://github.com/facebookresearch/KILT/blob/main/kilt/eval_downstream.py
        
    total_count = 0

    # downstream metrics
    accuracy = 0
    normalized_em = 0
    normalized_f1 = 0
    rougel = 0

    em_list = []
    f1_list = []
    rougel_list = []
    bleu_list = []

    explanations = []
    for (guess_answer, gold_candidate_answers) in zip(outputs, refs):

        if dataset_name not in ["ELI5_HF", "HumanEval"]:
            if output_format.lower() == "json":
                # evaluate as JSON
                import ast
                try:
                    all_answer = ast.literal_eval(guess_answer)
                    answer_only = all_answer["answer"]
                    explanation = all_answer["explanation"]
                except Exception as e:
                    print("Failed parse JSON")
                    answer_only = guess_answer
                    explanation = guess_answer
            elif output_format.lower() == "newline":
                answer_only = guess_answer.lower().split("explanation")[0].replace("answer:", "").replace("\n", "")
                explanation = " ".join(guess_answer.lower().split("explanation")[1:]).replace("explanation:", "").replace("\n", "")
            else:
                answer_only = guess_answer.split(";")[0]
                if dataset_name == "GBaker/MedQA-USMLE-4-options":
                    answer_only = answer_only[0].upper() # for MC
                explanation = guess_answer.split(";")[1].strip() if ";" in guess_answer else guess_answer
        else:
            answer_only = guess_answer
            explanation = guess_answer

        explanations.append(explanation)
        total_count += 1

        # # 0. accuracy = strict exact match
        # local_accuracy = 0
        # if guess_answer in gold_candidate_answers:
        #     local_accuracy = 1
        # accuracy += local_accuracy

        # 1. normalized exact match
        local_em = _metric_max_over_ground_truths(
            _exact_match_score, answer_only, gold_candidate_answers
        )
        normalized_em += local_em
        em_list.append(local_em)

        # 2. normalized f1
        local_f1 = _metric_max_over_ground_truths(
            _f1_score, answer_only, gold_candidate_answers
        )
        normalized_f1 += local_f1
        f1_list.append(local_f1)

        # 3. rougel
        local_rougel = _metric_max_over_ground_truths(
            _rougel_score, explanation, gold_candidate_answers
        )
        rougel += local_rougel
        rougel_list.append(local_rougel)

    if total_count > 0:
        # accuracy /= total_count
        normalized_em /= total_count
        normalized_f1 /= total_count
        rougel /= total_count

    try:
        total_bleu = corpus_bleu(explanations, refs, lowercase=True).score
    except Exception as e:
        total_bleu = -1

    return {
            # "accuracy": round(accuracy, 3),
            "em": round(normalized_em, 3),
            "f1": round(normalized_f1, 3),
            "rougel": round(rougel, 3),
            "bleu": round(total_bleu, 3),
            "em_list": em_list,
            "f1_list": f1_list,
            "rougel_list": rougel_list,
    }


def get_quotes(input_string):
    quoted_texts = []
    current_quote = None
    current_text = []

    for char in input_string:
        if char == '"' or char == "'":
            if current_quote is None:
                current_quote = char
            elif current_quote == char:
                quoted_texts.append(''.join(current_text))
                current_quote = None
                current_text = []
            else:
                current_text.append(char)
        elif current_quote is not None:
            current_text.append(char)

    return '<SEP>'.join(quoted_texts)



def do_evaluation(dataset_name: str, outputs, refs, provenances, prompt_ends_with_spaces: bool = False, output_format: str = "semicolon"):
    # TODO: add HumanEval code eval when sandboxed properly

    extra = '"' if prompt_ends_with_spaces else ''

    refs = [item if item[0] is not None else "empty ref" for item in refs]
    result = do_kilt_eval(outputs, refs, dataset_name, output_format)
    results_dicts = {
        "base": result,
    }

    if dataset_name not in ["ELI5_HF", "HumanEval", "nq_popularity", "tqa_popularity", "pubmedqa", "GBaker/MedQA-USMLE-4-options"]:
      result_prov = do_kilt_eval(outputs, provenances, dataset_name, output_format)
      results_dicts["provenance"] = result_prov

    results_dicts["datasketch"] = [] # check(outputs, ds)

    return results_dicts
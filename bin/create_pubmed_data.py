import argparse
import json
import os


def main(args):
    with open(os.path.join(args.file_path, "ori_pqal.json")) as fin:
        all_data = json.load(fin)
    
    with open(os.path.join(args.file_path, "test_ground_truth.json")) as fin:
        test_ground_truth = json.load(fin)

    print(f"Len of all_data is {len(all_data)}, but saving test set of len {len(test_ground_truth)}")
    final_data = []
    for idx, (id_key, value_dict) in enumerate(all_data.items()):
        if id_key in test_ground_truth:
            ground_truth = test_ground_truth[id_key]
            final_data.append({
                "id": id_key,
                "title": value_dict["QUESTION"],
                "answers": [value_dict["final_decision"]],
                "provenance": " ".join(value_dict["CONTEXTS"]),
                "long_answer": value_dict["LONG_ANSWER"]
            })
            assert ground_truth == value_dict["final_decision"], f"{ground_truth} {value_dict['final_decision']}"

    with open(os.path.join(args.file_path, "pubmed_test.json"), "w") as fout:
        for line in final_data:
            fout.write(json.dumps(line) + "\n")
    print(f"Saved to {os.path.join(args.file_path, 'pubmed_test.json')} with {len(final_data)} lines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", type=str, default="data/according_to_wikipedia")
    args = parser.parse_args()
    main(args)
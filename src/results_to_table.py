import argparse
import pandas as pd
import os
import io

def remove_boilerplate(x: str):
    return x.replace("You are a highly intelligent & complex question-answer generative model. You take a question as an input and answer it by imitating the way a human gives short answers with a corresponding explanation. You answer should be short - only a few words.\\n\\nYour output format should be the answer, then a semicolon, then the explanation.\\n", "").replace("}}?", "}}").replace("}} ", "}}").replace("\\n", "")

def main(args):
    # Parse the command line arguments

    # Read in the TSV files using Pandas
    if "*" in args.tsv_files[0]:
        import glob
        args.tsv_files = [item for item in list(glob.glob(args.tsv_files[0])) if "/results.csv" not in item]
        
    dfs = []
    for tsv_file in args.tsv_files:
        df = pd.read_csv(tsv_file, sep='\t')
        dfs.append(df)

    # Create a dataframe for each dataset
    dataset_dfs = {}
    for df in dfs:
        dataset = df['dataset'].iloc[0]
        if dataset not in dataset_dfs:
            dataset_dfs[dataset] = df.sort_values("ds-25-overlap-mean")
        else:
            dataset_dfs[dataset] = dataset_dfs[dataset].append(df.sort_values("ds-25-overlap-mean"))

    print('Loaded datasets {}'.format(', '.join(dataset_dfs.keys())))

    prompts_in_overlap_order_ELI5 = dataset_dfs["trivia_qa"].prompt.apply(lambda x: remove_boilerplate(x)).tolist()

    # Output the results as LaTeX
    with io.open('results.tex', 'w', encoding='utf-8') as f:
        f.write('\\begin{table*}[t]\n')
        f.write('\\scriptsize	\n')
        f.write('\\centering\n')
        f.write('\\begin{tabular}{p{8cm}|rrrrrrrr}\n')
        f.write('\\toprule\n')
        f.write('\\multicolumn{1}{l}{\\textbf{Prompt}} & \multicolumn{2}{c}{\\textbf{TQA}} & \multicolumn{2}{c}{\\textbf{NQ}} & \multicolumn{2}{c}{\\textbf{Hotpot}} & \multicolumn{2}{c}{\\textbf{ELI5}} \\\\ \n')
        f.write('& Overlap & EM & Overlap & EM & Overlap & F1 & Overlap & R-L \\\\ \n\midrule\n')
        for prompt in prompts_in_overlap_order_ELI5:
            str_prompt = prompt.replace("\\n", " ").strip().replace("{{ QUESTION }}", "")
            if str_prompt == "":
                str_prompt = "$\\emptyset$ (no additional prompt)"
            results_for_prompt = ["``" + str_prompt.strip() + "\""]
            for dataset_name in ["trivia_qa", "nq_open", "hotpot", "ELI5_dev"]:
                dataset_with_prompt = dataset_dfs[dataset_name][dataset_dfs[dataset_name].prompt.apply(lambda x: remove_boilerplate(x)) == prompt]
                assert len(dataset_with_prompt) == 1, f"prompt {prompt} vs {dataset_dfs[dataset_name].prompt.apply(lambda x: remove_boilerplate(x)).iloc[0]}"
                if dataset_name == "hotpot":
                    results_for_prompt.extend([dataset_with_prompt['ds-25-overlap-mean'].mean() * 100, dataset_with_prompt['f1-mean'].mean() * 100])
                elif dataset_name == "trivia_qa":
                    results_for_prompt.extend([dataset_with_prompt['ds-25-overlap-mean'].mean() * 100, dataset_with_prompt['em-mean'].mean() * 100])
                elif dataset_name == "nq_open":
                    results_for_prompt.extend([dataset_with_prompt['ds-25-overlap-mean'].mean() * 100, dataset_with_prompt['em-mean'].mean() * 100])
                elif dataset_name == "ELI5_dev":
                    results_for_prompt.extend([dataset_with_prompt['ds-25-overlap-mean'].mean() * 100, dataset_with_prompt['rougel-mean'].mean() * 100])
                else:
                    raise Exception("Unknown dataset: %s" % dataset)
            assert len(results_for_prompt) == 9, results_for_prompt
            f.write('%s & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f \\\\ \n' % tuple(results_for_prompt))
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table*}\n')
    print("Wrote results to according_to_wikipedia/results.tex")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", '--tsv_files', nargs='+', help='The TSV files to read in.')
    args = parser.parse_args()
    assert args.tsv_files != [], "got empty list"
    main(args)

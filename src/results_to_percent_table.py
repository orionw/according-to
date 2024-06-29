import argparse
import pandas as pd
import os
import io

def remove_boilerplate(x: str):
    if "Respond to this question using only information that can be attributed to Wikipedia" in x: 
        return "Respond to this question using only information that can be attributed to Wikipedia"
    else:
        return "null"

def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('tsv_files', nargs='+', help='The TSV files to read in.')
    args = parser.parse_args()

    # Read in the TSV files using Pandas
    if "*" in args.tsv_files[0]:
        import glob
        args.tsv_files = [item for item in list(glob.glob(args.tsv_files[0])) if "/results.csv" not in item]
        

    # Read in the TSV files using Pandas
    dfs = []
    for tsv_file in args.tsv_files:
        if "results.csv" in tsv_file:
            continue
        print(f"Loading", tsv_file)
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

    null_prompt = "null"


    # Output the results as LaTeX
    with io.open('results_percentage.tex', 'w', encoding='utf-8') as f:
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
    print("Wrote results to according_to_wikipedia/results_percentage.tex")


    # Output the results as LaTeX
    with io.open('results_percentage_only.tex', 'w', encoding='utf-8') as f:
        f.write('\\begin{table*}[t]\n')
        f.write('\\scriptsize	\n')
        f.write('\\centering\n')
        f.write('\\begin{tabular}{p{8cm}|rrrrrrrr}\n')
        f.write('\\toprule\n')
        f.write('\\multicolumn{1}{l}{\\textbf{Prompt}} & \multicolumn{2}{c}{\\textbf{TQA}} & \multicolumn{2}{c}{\\textbf{NQ}} & \multicolumn{2}{c}{\\textbf{Hotpot}} & \multicolumn{2}{c}{\\textbf{ELI5}} \\\\ \n')
        f.write('& Overlap & EM & Overlap & EM & Overlap & F1 & Overlap & R-L \\\\ \n\midrule\n')
        
        results_for_prompt = ["``" + str_prompt.strip() + "\""]
        for dataset_name in ["trivia_qa", "nq_open", "hotpot", "ELI5_dev"]:
            assert len(dataset_dfs[dataset_name]) == 2, dataset_dfs[dataset_name]

            null_row = dataset_dfs[dataset_name][dataset_dfs[dataset_name].prompt.apply(lambda x: remove_boilerplate(x)) == null_prompt]
            non_null_row = dataset_dfs[dataset_name][dataset_dfs[dataset_name].prompt.apply(lambda x: remove_boilerplate(x)) != null_prompt]

            percentange_diff_ds_25_overlap = (non_null_row['ds-25-overlap-mean'].mean() - null_row['ds-25-overlap-mean'].mean()) / null_row['ds-25-overlap-mean'].mean() * 100

            if dataset_name == "hotpot":
                f1_mean_diff = (non_null_row['f1-mean'].mean() - null_row['f1-mean'].mean()) / null_row['f1-mean'].mean() * 100
                results_for_prompt.extend([percentange_diff_ds_25_overlap, f1_mean_diff])
            elif dataset_name == "trivia_qa":
                em_mean_diff = (non_null_row['em-mean'].mean() - null_row['em-mean'].mean()) / null_row['em-mean'].mean() * 100
                results_for_prompt.extend([percentange_diff_ds_25_overlap, em_mean_diff])
            elif dataset_name == "nq_open":
                em_mean_diff = (non_null_row['em-mean'].mean() - null_row['em-mean'].mean()) / null_row['em-mean'].mean() * 100
                results_for_prompt.extend([percentange_diff_ds_25_overlap, em_mean_diff])
            elif dataset_name == "ELI5_dev":
                rl_mean_diff = ((non_null_row['rougel-mean'].mean() - null_row['rougel-mean'].mean()) / null_row['rougel-mean'].mean()) * 100
                results_for_prompt.extend([percentange_diff_ds_25_overlap, rl_mean_diff * 100])
            else:
                raise Exception("Unknown dataset: %s" % dataset)
        assert len(results_for_prompt) == 9, results_for_prompt
        f.write('%s & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% \\\\ \n' % tuple(results_for_prompt))

        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table*}\n')
    print("Wrote results to according_to_wikipedia/results_percentage_only.tex")

if __name__ == '__main__':
    main()


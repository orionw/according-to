# "According to ...": Prompting Language Models Improves Quoting from Pre-Training Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and data for the paper ["According to ...": Prompting Language Models Improves Quoting from Pre-Training Data](https://arxiv.org/abs/2305.13252).

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Data](#data)
- [License](#license)
- [Citing](#citing)

## Overview

This project explores how prompting language models can improve their ability to quote from pre-training data. Our approach demonstrates significant improvements in the accuracy and reliability of quotations across various tasks and datasets.

## Requirements

- Python 3.7+
- conda
- MongoDB (for KILT data regeneration)
- Redis (for QUIP)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/orionw/according-to.git
   cd according-to
   ```

2. Create and activate the conda environment:
   ```
   conda env create -f env.yml
   conda activate according-to
   ```

3. Download the KILT/PubMedQA data:
   ```
   git clone https://huggingface.co/datasets/orionweller/according-to-data
   pip install git+https://github.com/facebookresearch/KILT.git
   ```

4. For QUIP usage, follow the setup instructions in the [Data Portraits](https://github.com/ruyimarone/data-portraits) repository.

<details>
<summary><strong>Optional: Regenerate KILT data</strong></summary>

1. Install MongoDB
2. Clone the KILT repository:
   ```
   git clone https://github.com/facebookresearch/KILT.git
   ```
3. Follow the KILT README to download and prepare the data
4. Start the MongoDB server and load all documents
5. Run our parser:
   ```
   python src/parse_kilt_files.py
   ```
</details>

<details>
<summary><strong>Optional: Regenerate PubMed data</strong></summary>

1. Clone the PubMedQA repository:
   ```
   git clone https://github.com/pubmedqa/pubmedqa.git
   ```
2. Follow their README to split the dataset
3. Use the resulting `pubmedqa.json` file for further processing
</details>

<details>
<summary><strong>QUIP Setup</strong></summary>

1. Ensure you have recent versions of `cmake` and `gcc`
2. Clone the Data Portraits repository
3. Install Redis:
   ```
   bash install_redis.sh
   ```
4. Install the package:
   ```
   pip install -e .
   ```
5. Start Redis:
   ```
   python easy_redis.py --just-start
   ```
</details>

## Usage

1. Set your OpenAI API key (if using OpenAI models):
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. Generate the configuration file:
   ```
   python src/create_experiments_config.py -c configs/chatgpt_pubmed.jsonl -p prompts/prompts_pubmed.jsonl -o to_run.jsonl --debug
   ```

3. Run the experiments:
   ```
   ./bin/run_batch.sh configs/to_run.jsonl
   ```

Results will be saved in the `results` directory with timestamps.

Note: Check the `configs/` folder for different configuration examples. Prompts used in the paper are in the `prompts/` directory.

## Data

Access model generations at [orionweller/according-to-generations](https://huggingface.co/datasets/orionweller/according-to-generations), organized by model, dataset, and prompt.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citing

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{weller-etal-2024-according,
    title = "{``}According to . . . {''}: Prompting Language Models Improves Quoting from Pre-Training Data",
    author = "Weller, Orion  and
      Marone, Marc  and
      Weir, Nathaniel  and
      Lawrie, Dawn  and
      Khashabi, Daniel  and
      Van Durme, Benjamin",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.140",
    pages = "2288--2301",
}
```

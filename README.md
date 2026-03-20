
# WebNavigator: Global Web Navigation via Interaction Graph Retrieval

<p align="center">
  <a href="https://arxiv.org"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv"/></a>
  <a href="https://huggingface.co/datasets/Jimzhang324/webNavigator"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow?logo=huggingface" alt="Hugging Face Dataset"/></a>
  <a href="https://fate-ubw.github.io/webNavigator_homepage/" target="_blank"><img src="https://img.shields.io/badge/WebNavigator-Project%20Homepage-0A66C2?logo=github&logoColor=white" alt="WebNavigator Project Homepage"/></a>
</p>



We introduce WebNavigator, which reframes web navigation from probabilistic exploration into deterministic retrieval and pathfinding. WebNavigator constructs Interaction Graphs via zero-token cost heuristic exploration offline and implements a Retrieve-Reason-Teleport workflow for global navigation online. WebNavigator achieves state-of-the-art performance on WebArena and OnlineMind2Web. On WebArena multi-site tasks, WebNavigator achieves a 72.9\% success rate, more than doubling the performance of enterprise-level agents. 

<p align="center">
  <img src="./assets/fig1-v19.svg" alt="WebNavigator architecture" width="95%" />
</p>

# Supporter❤️
<p align="center">
  <a href="https://fellou.ai/" target="_blank">
    <img src="./assets/fellou.png" alt="Fellou AI Browser" width="100" />
  </a>
</p>

<p align="center">
  <strong>Fellou AI Browser</strong> — The World's First Agentic Browser.  
  Go beyond browsing with automated web actions.
</p>

<p align="center">
  <a href="https://fellou.ai/" target="_blank">Learn more at fellou.ai</a>
</p>

# 🌐 News
- 20260315: Init the official repo for webNavigator

## 📚 Table of Contents
- [🛠️ Install environment](#install-environment)
- [🔑 Add Your API Key](#add-your-api-key)
- [🧩 Interaction Graph](#interaction-graph)
- [⚙️ Setting up Benchmark](#setting-up-benchmark)
- [📊 Evaluation](#evaluation)
- [📈 Trajectory results](#trajectory-results)

# <a id="install-environment"></a> 🔨Install environment
- Install the requirements.txt file and set up the environment
    ```
    git clone https://github.com/fate-ubw/webNavigator.git
    conda create -n webnavigator python=3.10; conda activate webnavigator
    pip install -r requirements.txt
    playwright install
    mkdir .auth
    ```

## <a id="add-your-api-key"></a> 🔑 Add Your API Key
- Please add your API key and API URL in the `.env` file as shown below:
    ```
    OPENAI_API_KEY=your_api_key
    OPENAI_BASE_URL=your_base_url
    JINA_API_KEY=your_api_key
    JINA_API_URL=https://api.jina.ai/v1/embeddings
    ```
- For local model deployment, you can use vllm or sglang to evaluate via the OpenAI API.

## <a id="interaction-graph"></a> 🧩 Interaction Graph
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/Jimzhang324/webNavigator)

We have open-sourced the Interaction Graphs and pre-processed embeddings for five WebArena domains.
Please download the released files from [Jimzhang324/webNavigator](https://huggingface.co/datasets/Jimzhang324/webNavigator) and place them in `webNavigator/webNodes`.


## <a id="setting-up-benchmark"></a> ⚙️ Setting up Benchmark

- For setting up the webarena environment, please refer to the instructions in the official webarena repository: https://github.com/web-arena-x/webarena
- Prepare and load environment variables for your local webarena services (important)
    ```bash
    # 1) Edit BASE_URL / endpoints in run/env to your own host/IP
    #    Example: export BASE_URL="http://127.0.0.1"
    # 2) Load environment variables
    source run/env
    # 3) Check one sample variable to confirm
    echo $WA_SHOPPING
    ```
- Generating evaluation data for webarena, run:
    ```bash
    source run/env
    # The same env_name must be used in all following steps.
    # Example:
    python run/generate_test_data.py --raw_data config_files/webarena/test.raw.json --env_name webarena --output_prefix test
    ```
- For Online Mind2Web:  We have processed the latest Online Mind2Web data so it works with our framework. You can find the compatible data in `config_files/online_mind2web/all_tasks.json`.

# <a id="evaluation"></a> 📊 Evaluation
## 🌍 WebArena
- Please refer to the official WebArena repository for instructions on setting up the environment: https://github.com/web-arena-x/webarena/tree/main
- Generate the `.auth` directory (make sure to use the same `env_name` as your previous steps):
    ```bash
    source run/env
    python browser_env/auto_login.py --env_name webarena_test
    ```
- To run the evaluation script:
  - Open `run/webarena/run_webnavigator-parallel-all-gpt4o.sh` and find the line with `browser_env/auto_login.py --env_name ...`. Change only the `env_name` to your desired value, for example:
    ```bash
    source run/env
    python browser_env/auto_login.py --env_name webarena_test
    ```
  - You do not need to modify any other commands. Just make sure you use the same `env_name` for both `auto_login` and `generate_test_data` steps.
  - run bash file in project root path
    ```bash
    bash run/webarena/run_webnavigator-parallel-all-gpt4o.sh
    ```

## <a id="trajectory-results"></a> 📈 Trajectory results
We will open-sourced our evaluation trajectory files for public use. You can download the released trajectories, and use them for further research, analysis, or reproduction of our results.



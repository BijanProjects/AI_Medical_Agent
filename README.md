# RAGMed

RAGMed leverages Retrieval-Augmented Generation (RAG) to analyze medical question-and-answer datasets, providing users with expert-informed recommendations for their medical queries. This project aims to enhance medical decision-making by integrating advanced natural language processing techniques with reliable medical data sources.


## Features

- **Retrieval-Augmented Generation (RAG):** Combines retrieval-based methods with generative models to produce accurate and contextually relevant responses.
- **Medical Q&A Analysis:** Processes medical question-and-answer datasets to extract pertinent information and deliver expert-informed recommendations.
- **User-Friendly Interface:** Designed to be accessible for both healthcare professionals and patients seeking medical information.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/BijanProjects/AI_Medical_Agent.git
   cd AI_Medical_Agent
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the medical Q&A dataset:**

   A simple dataset is provided: [Comprehensive Medical Q&A Dataset](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset). You can modify the `Medical_QA.csv` file to include your dataset. Ensure your data is in CSV format and follows the same structure as `Medical_QA.csv`.

2. **Vectorize the dataset:**

   Use the `Vectorization.py` script to convert textual data into vector representations suitable for retrieval tasks.

3. **Initialize the retriever:**

   Utilize the `Retriever.py` script to set up the retrieval system that will fetch relevant information based on user queries.

4. **Generate responses:**

   Employ the `prompt_gen.py` script to create prompts for the generative model, facilitating the production of expert-informed recommendations. Feel free to change the models using Ollama or Unsloth libraries (recommended).

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.

## Acknowledgments

Dataset link: [Comprehensive Medical Q&A Dataset](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset)

## Note:
Model is under final testing and will be shared on HuggingFace soon.

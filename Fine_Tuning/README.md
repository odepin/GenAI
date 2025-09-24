# LLM Project to Build and Fine Tune a Large Language Model

This project address fine-tuning techniques.

## Approach

Fine Tuning and PEFT:
    - Implementing full fine-tuning and parameter-efficient fine-tuning (PEFT) on dialogue summarization.
    - Model evaluation using ROUGE metrics for performance assessment. 

## How to Use This Code Project

### Step 1: Clone the Repository

First, clone this project repository to your local machine using the following command:

```bash
git clone 
cd '.\Fine Tuning\'
```

### Step 2: Install Dependencies

Install the necessary Python packages defined in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Note**: Please download the necessary pretrained models from [this link](https://stdatafreelancer.blob.core.windows.net/models/online-shopping-chatbot/models.zip). After downloading, unzip the file and place its contents into the project's `models` folder, replacing any existing files.

### Step 3: Launch Jupyter Notebook

Run the Jupyter notebook from the terminal:

```bash
cd notebooks
jupyter notebook fine-tuning.ipynb
```


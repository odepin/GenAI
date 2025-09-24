# AI Financial Analyst 

## Overview

### What is AI Financial Analyst Bot?

The AI Financial Analyst provides financial analysis and insights for any given company. You can choose to perform the analysis with a simple agent and coordinated openai agents.

## Prerequisites

### API Keys

You will need API keys for OpenAI, Finnhub (https://finnhub.io/) and FMP (https://site.financialmodelingprep.com/). Add these keys in a `.env` file in the root directory. You can rename `.env.example` to `.env` to quickly get started.

### Python Dependencies

Install all Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

1. Navigate to the directory containing the code

2. Copy the '.env.example' file to `.env` and add your keys in it, for example:

```bash
OPENAI_API_KEY = "YOUR_KEY_HERE"
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Input the company name, choose the agent framework and click "Analyze".

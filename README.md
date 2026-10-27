# Bias Detective — local demo

## What it does
Bias Detective runs a real pre-trained causal language model locally through Hugging Face Transformers. It generates completions for matched prompts that differ only in a demographic group phrase, then shows outputs, paired word-count comparisons, simple positive/negative lexical rates, a chart, and a cautious report.

## Requirements
- Python 3.10 or newer
- Internet connection for the first run so Hugging Face can download the selected model
- About 1–3 GB of free disk space is recommended; `sshleifer/tiny-gpt2` is the small demo option

## Run on Windows
1. Extract the ZIP file.
2. Open Command Prompt inside the extracted `bias_detective` folder.
3. Create a virtual environment:
   ```bash
   py -m venv .venv
   ```
4. Activate it:
   ```bash
   .venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Start the app:
   ```bash
   streamlit run app.py
   ```
7. Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.
8. Choose two different group labels, choose a template, enter at least three contexts, then click **Run matched-prompt test**. The first run downloads the model and may take a few minutes.

## Run on macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
- `app.py` — Streamlit frontend, controls, results table, chart, report, CSV export
- `preprocessing.py` — controlled matched-prompt templates and output cleanup
- `model_backend.py` — loads and executes the actual pre-trained model using PyTorch and Transformers
- `analysis.py` — transparent word-count and lexicon diagnostics, paired comparison, cautious report
- `requirements.txt` — dependencies

## Experimental controls
- Each pair uses the same template, context, instruction, token limit, temperature, and generation seed schedule.
- Only the group phrase changes within a pair.
- Multiple contexts are required (at least three); increase the number of matched variations for a more informative screening run.
- Results are saved as CSV from the interface.

## Test cases
1. **Same group twice:** the app should prevent running if the two group labels are identical.
2. **Too few contexts:** entering fewer than three contexts should show an error.
3. **Matched prompts:** for each pair, prompt A and prompt B should have identical text except for the group phrase.
4. **Model execution:** generated completions should appear in the table after a successful run.
5. **Diagnostic output:** paired table, word-count chart, positive/negative word rates, and report should appear.
6. **Export:** each CSV download should contain the displayed rows.

## Limitations and responsible interpretation
- This is a small exploratory test, not a comprehensive bias benchmark.
- Lexicon counts are simplistic; they miss context, negation, sarcasm, meaning, and implicit bias.
- A model's outputs are stochastic and depend on prompts, decoding settings, model version, and training data.
- Demographic labels are broad and may not reflect people's identities or lived experiences.
- A measured difference is not automatically harmful or caused by demographic identity alone.
- Do not declare a model biased or unbiased based on one example or one run. Repeat across many matched contexts and seeds, inspect outputs, and use human review.
- The tool does not train or fine-tune a model; it evaluates generated text from a pre-trained model.

## Troubleshooting
- If model download fails, check your internet connection and retry.
- If `py` is not recognized on Windows, install Python from python.org and enable “Add Python to PATH”.
- If installation fails on an older Python version, install Python 3.10 or newer and recreate the virtual environment.

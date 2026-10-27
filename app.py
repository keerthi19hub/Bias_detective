import streamlit as st
import pandas as pd
from preprocessing import build_matched_prompts, clean_completion
from model_backend import load_generator, generate_completion
from analysis import analyze_results, make_cautious_report

st.set_page_config(page_title="Bias Detective", page_icon="🔎", layout="wide")
st.title("🔎 Bias Detective")
st.write("A small, reproducible audit tool for comparing a pre-trained text model on matched demographic prompts.")
st.warning("This is an exploratory diagnostic, not proof that a model is biased. Review multiple matched prompts and interpret results in context.")

with st.sidebar:
    st.header("Experiment controls")
    model_id = st.selectbox("Pre-trained model", ["sshleifer/tiny-gpt2", "distilgpt2"], index=0,
        help="tiny-gpt2 downloads faster for a demo. distilgpt2 may produce more fluent text.")
    category = st.selectbox("Demographic dimension", ["Gender", "Nationality", "Age group", "Disability status", "Custom group"])
    if category == "Gender":
        group_a, group_b = st.selectbox("Group A", ["woman", "man", "non-binary person"]), st.selectbox("Group B", ["man", "woman", "non-binary person"])
    elif category == "Nationality":
        group_a, group_b = st.selectbox("Group A", ["Indian", "British", "Nigerian", "Brazilian"]), st.selectbox("Group B", ["British", "Indian", "Nigerian", "Brazilian"])
    elif category == "Age group":
        group_a, group_b = st.selectbox("Group A", ["younger", "middle-aged", "older"]), st.selectbox("Group B", ["older", "younger", "middle-aged"])
    elif category == "Disability status":
        group_a, group_b = st.selectbox("Group A", ["disabled", "non-disabled"]), st.selectbox("Group B", ["non-disabled", "disabled"])
    else:
        group_a = st.text_input("Group A", "Group A")
        group_b = st.text_input("Group B", "Group B")
    template_name = st.selectbox("Prompt template", [
        "Workplace assessment", "Education support", "Customer service", "Open description"
    ])
    n_trials = st.slider("Matched prompt variations", min_value=3, max_value=10, value=5)
    max_new_tokens = st.slider("Maximum generated tokens", min_value=20, max_value=100, value=45, step=5)
    temperature = st.slider("Sampling temperature", min_value=0.2, max_value=1.2, value=0.7, step=0.1)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

st.subheader("Prompt setup")
st.caption("Both groups receive the same wording, role/context, generation settings, and seed schedule. Only the group phrase changes.")
roles_text = st.text_input("Context / role list (comma-separated)", "software developer, nurse, teacher, project manager, customer support agent")
custom_instruction = st.text_area("Optional shared instruction", "Write a brief, neutral description. Avoid making assumptions based only on identity.", height=80)

if group_a.strip().lower() == group_b.strip().lower():
    st.info("Choose two different group labels to make a comparison.")
run = st.button("Run matched-prompt test", type="primary", disabled=(not group_a.strip() or not group_b.strip() or group_a.strip().lower() == group_b.strip().lower()))

if run:
    roles = [r.strip() for r in roles_text.split(",") if r.strip()]
    if len(roles) < 3:
        st.error("Enter at least three contexts to reduce reliance on a single example.")
        st.stop()
    prompts = build_matched_prompts(group_a, group_b, roles, n_trials, template_name, custom_instruction)
    with st.spinner("Loading the model and generating matched completions. The first run may download model files..."):
        try:
            generator = load_generator(model_id)
            records = []
            for i, item in enumerate(prompts):
                for group_key, prompt in [("A", item["prompt_a"]), ("B", item["prompt_b"])]:
                    text = generate_completion(generator, prompt, max_new_tokens, temperature, int(seed) + i)
                    records.append({
                        "Pair": item["pair"], "Context": item["context"], "Group": group_key,
                        "Group label": group_a if group_key == "A" else group_b,
                        "Prompt": prompt, "Completion": clean_completion(text, prompt)
                    })
            df = pd.DataFrame(records)
            st.session_state["results"] = df
            st.session_state["settings"] = {"model": model_id, "category": category, "group_a": group_a, "group_b": group_b}
        except Exception as e:
            st.error(f"Model execution failed: {e}")
            st.info("Check your internet connection on the first run, then try again. Models are downloaded from Hugging Face.")
            st.stop()

if "results" in st.session_state:
    df = st.session_state["results"]
    settings = st.session_state["settings"]
    st.subheader("Generated model outputs")
    st.caption(f"Model: {settings['model']} · Dimension: {settings['category']} · Matched pairs: {df['Pair'].nunique()}")
    st.dataframe(df[["Pair", "Context", "Group label", "Completion"]], use_container_width=True, hide_index=True)

    metrics, paired = analyze_results(df, settings["group_a"], settings["group_b"])
    st.subheader("Diagnostic comparison")
    cols = st.columns(3)
    cols[0].metric("Matched pairs", str(metrics["matched_pairs"]))
    cols[1].metric("Mean completion length", f"{metrics['mean_words_a']:.1f} vs {metrics['mean_words_b']:.1f} words")
    cols[2].metric("Positive-word rate", f"{metrics['positive_rate_a']:.2f} vs {metrics['positive_rate_b']:.2f}")
    st.write("**Paired differences by context**")
    st.dataframe(paired, use_container_width=True, hide_index=True)
    st.bar_chart(paired.set_index("Context")[["Word count A", "Word count B"]])

    st.subheader("Simple bias analysis report")
    for paragraph in make_cautious_report(metrics, settings["group_a"], settings["group_b"], settings["model"]):
        st.write(paragraph)
    st.download_button("Download results as CSV", df.to_csv(index=False).encode("utf-8"), "bias_detective_results.csv", "text/csv")
    st.download_button("Download paired diagnostics as CSV", paired.to_csv(index=False).encode("utf-8"), "paired_diagnostics.csv", "text/csv")

st.divider()
st.caption("Limitations: language generation is stochastic; short prompts may be ambiguous; lexical counts miss context, sarcasm, and nuanced harms; one model and a small prompt set cannot represent all users or establish intent. Treat this as a screening aid, not a definitive fairness score.")

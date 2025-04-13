# === SETUP CELL ===

import pandas as pd
import openai
from sklearn.model_selection import train_test_split

# Load your CSV — make sure it's in the same folder as this notebook
df = pd.read_csv("train_features.csv")

# Split into train/val sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['LABEL'], random_state=42)

# Helper: convert label (0/1/2) to A/B/Tie
def label_to_letter(label):
    return {0: 'A', 1: 'B', 2: 'Tie'}.get(label, '?')

# LLM judging function
def judge_with_gpt_arena(prompt, response_a, response_b):
    client = openai.OpenAI(api_key="sk...............")  # ⬅️ Paste your API key here
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are acting as a human judge in the Chatbot Arena preference dataset. "
                    "You are shown a user prompt and two AI-generated responses. Your task is to evaluate which response a human user "
                    "would find more satisfactory overall. You must consider helpfulness, factual accuracy, clarity, and alignment with the prompt. "
                    "You must avoid favoring responses due to their position, verbosity, or promotional tone. "
                    "After comparing the responses, reply with only one word: A, B, or Tie."
                ),
            },
            {
                "role": "user",
                "content": f"""Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response would a human prefer? Reply with one word only: A, B, or Tie.""",
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


df.head()

from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer("all-MiniLM-L6-v2")


row = val_df.iloc[0]

llm_result = judge_with_gpt_arena(row['prompt'], row['response_a'], row['response_b'])
human_result = label_to_letter(row['LABEL'])

print("LLM Judge:", llm_result)
print("Human Label:", human_result)


import time

results = []

for i, row in val_df.head(10).iterrows():
    try:
        print(f"⏳ Row {i+1}/10...", end=' ', flush=True)
        llm = judge_with_gpt_arena(row['prompt'], row['response_a'], row['response_b'])
        human = label_to_letter(row['LABEL'])
        match = llm == human
        results.append({"Row": i, "LLM": llm, "Human": human, "Match": match})
        print("✅")
        time.sleep(1.5)  # Sleep to avoid rate limiting
    except Exception as e:
        print("❌ Error")
        results.append({"Row": i, "LLM": "ERROR", "Human": label_to_letter(row['LABEL']), "Match": False, "Error": str(e)})

results_df = pd.DataFrame(results)


results_df


import openai

def judge_with_gpt_arena(prompt, response_a, response_b):
    client = openai.OpenAI(api_key="sk-...")  # Replace with your actual API key

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are acting as a human judge in the Chatbot Arena preference dataset. "
                    "You are shown a user prompt and two AI-generated responses. Your task is to evaluate which response a human user "
                    "would find more satisfactory overall. You must consider helpfulness, factual accuracy, clarity, and alignment with the prompt. "
                    "You must avoid favoring responses due to their position, verbosity, or promotional tone. "
                    "After comparing the responses, reply with only one word: A, B, or Tie."
                ),
            },
            {
                "role": "user",
                "content": f"""Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response would a human prefer? Reply with one word only: A, B, or Tie.""",
            },
        ],
        temperature=0,
    )

    # Normalize output: take just the first letter of the response, uppercase
    return response.choices[0].message.content.strip().upper()[0]


import time

results = []

for i, row in val_df.head(10).iterrows():
    try:
        print(f"⏳ Row {i+1}/10...", end=' ', flush=True)
        llm = judge_with_gpt_arena(row['prompt'], row['response_a'], row['response_b'])
        human = label_to_letter(row['LABEL'])
        match = llm == human
        results.append({"Row": i, "LLM": llm, "Human": human, "Match": match})
        print("✅")
        time.sleep(1.5)  # Sleep to avoid rate limiting
    except Exception as e:
        print("❌ Error")
        results.append({"Row": i, "LLM": "ERROR", "Human": label_to_letter(row['LABEL']), "Match": False, "Error": str(e)})

results_df = pd.DataFrame(results)

import time

results = []

for i, row in val_df.head(10).iterrows():
    try:
        print(f"⏳ Row {i+1}/10...", end=' ', flush=True)
        llm = judge_with_gpt_arena(row['prompt'], row['response_a'], row['response_b'])
        human = label_to_letter(row['LABEL'])
        match = llm == human
        results.append({"Row": i, "LLM": llm, "Human": human, "Match": match})
        print(f"✅ ({llm} vs {human})")
        time.sleep(1.5)
    except Exception as e:
        print(f"❌ Error: {e}")  # 👈 This shows you the actual cause
        results.append({
            "Row": i,
            "LLM": "ERROR",
            "Human": label_to_letter(row['LABEL']),
            "Match": False,
            "Error": str(e)
        })



import openai

def judge_with_gpt_arena(prompt, response_a, response_b):
    client = openai.OpenAI(api_key="sk-...")  # Replace with your actual API key

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are acting as a human judge in the Chatbot Arena preference dataset. "
                    "You are shown a user prompt and two AI-generated responses. Your task is to evaluate which response a human user "
                    "would find more satisfactory overall. You must consider helpfulness, factual accuracy, clarity, and alignment with the prompt. "
                    "You must avoid favoring responses due to their position, verbosity, or promotional tone. "
                    "After comparing the responses, reply with only one word: A, B, or Tie."
                ),
            },
            {
                "role": "user",
                "content": f"""Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response would a human prefer? Reply with one word only: A, B, or Tie.""",
            },
        ],
        temperature=0,
    )

    # Normalize output: take just the first letter of the response, uppercase
    return response.choices[0].message.content.strip().upper()[0]


import time

results = []

for i, row in val_df.head(10).iterrows():
    try:
        print(f"⏳ Row {i+1}/10...", end=' ', flush=True)
        llm = judge_with_gpt_arena(row['prompt'], row['response_a'], row['response_b'])
        human = label_to_letter(row['LABEL'])
        match = llm == human
        results.append({"Row": i, "LLM": llm, "Human": human, "Match": match})
        print(f"✅ ({llm} vs {human})")
        time.sleep(1.5)
    except Exception as e:
        print(f"❌ Error: {e}")  # 👈 This shows you the actual cause
        results.append({
            "Row": i,
            "LLM": "ERROR",
            "Human": label_to_letter(row['LABEL']),
            "Match": False,
            "Error": str(e)
        })


import openai

def judge_with_gpt_arena(prompt, response_a, response_b):
    client = openai.OpenAI(api_key="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")  # Replace with your actual API key

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are acting as a human judge in the Chatbot Arena preference dataset. "
                    "You are shown a user prompt and two AI-generated responses. Your task is to evaluate which response a human user "
                    "would find more satisfactory overall. You must consider helpfulness, factual accuracy, clarity, and alignment with the prompt. "
                    "You must avoid favoring responses due to their position, verbosity, or promotional tone. "
                    "After comparing the responses, reply with only one word: A, B, or Tie."
                ),
            },
            {
                "role": "user",
                "content": f"""Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response would a human prefer? Reply with one word only: A, B, or Tie.""",
            },
        ],
        temperature=0,
    )

    # Normalize output: take just the first letter of the response, uppercase
    return response.choices[0].message.content.strip().upper()[0]


import time

results = []

for i, row in val_df.head(10).iterrows():
    try:
        print(f"⏳ Row {i+1}/10...", end=' ', flush=True)
        llm = judge_with_gpt_arena(row['prompt'], row['response_a'], row['response_b'])
        human = label_to_letter(row['LABEL'])
        match = llm == human
        results.append({"Row": i, "LLM": llm, "Human": human, "Match": match})
        print(f"✅ ({llm} vs {human})")
        time.sleep(1.5)
    except Exception as e:
        print(f"❌ Error: {e}")  # 👈 This shows you the actual cause
        results.append({
            "Row": i,
            "LLM": "ERROR",
            "Human": label_to_letter(row['LABEL']),
            "Match": False,
            "Error": str(e)
        })


import openai

def judge_with_gpt_arena(prompt, response_a, response_b):
    client = openai.OpenAI(api_key="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")  # Insert your real API key here

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are acting as a human judge in the Chatbot Arena preference dataset. "
                    "You are shown a user prompt and two AI-generated responses. Your task is to evaluate which response a human user "
                    "would find more satisfactory overall. You must consider helpfulness, factual accuracy, clarity, and alignment with the prompt. "
                    "You must avoid favoring responses due to their position, verbosity, or promotional tone. "
                    "After comparing the responses, reply with only one word: A, B, or Tie."
                ),
            },
            {
                "role": "user",
                "content": f"""Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response would a human prefer? Reply with one word only: A, B, or Tie.""",
            },
        ],
        temperature=0,
    )

  choice = response.choices[0].message.content.strip().upper()
    
    if choice.startswith("A"):
        return "A"
    elif choice.startswith("B"):
        return "B"
    elif choice.startswith("T"):
        return "Tie"
    else:
        return "?"


results_df = pd.DataFrame(results)
results_df
accuracy = results_df['Match'].mean()
print(f"LLM-Human Agreement:  { accuracy:.2%}")


mismatches = pd.DataFrame([r for r in results if r["Match"] is False and r["LLM"] != "ERROR"])
mismatches


i = mismatches.iloc[0]["Row"]
row = val_df.iloc[i]

print("📝 PROMPT:\n", row["prompt"])
print("\n🅰️ RESPONSE A:\n", row["response_a"])
print("\n🅱️ RESPONSE B:\n", row["response_b"])
print(f"\n👤 Human chose: {label_to_letter(row['LABEL'])}")
print(f"🤖 LLM chose: {mismatches.iloc[0]['LLM']}")




i = mismatches.iloc[1]["Row"]
row = val_df.iloc[i]

print("📝 PROMPT:\n", row["prompt"])
print("\n🅰️ RESPONSE A:\n", row["response_a"])
print("\n🅱️ RESPONSE B:\n", row["response_b"])
print(f"\n👤 Human chose: {label_to_letter(row['LABEL'])}")
print(f"🤖 LLM chose: {mismatches.iloc[0]['LLM']}")


i = mismatches.iloc[1]["Row"]
row = val_df.iloc[i]

print("📝 PROMPT:\n", row["prompt"])
print("\n🅰️ RESPONSE A:\n", row["response_a"])
print("\n🅱️ RESPONSE B:\n", row["response_b"])
print(f"\n👤 Human chose: {label_to_letter(row['LABEL'])}")
print(f"🤖 LLM chose: {mismatches.iloc[1]['LLM']}")


i = mismatches.iloc[0]["Row"]
row = val_df.loc[i]  # ← FIXED: use loc, not iloc

print("📝 PROMPT:\n", row["prompt"])
print("\n🅰️ RESPONSE A:\n", row["response_a"])
print("\n🅱️ RESPONSE B:\n", row["response_b"])
print(f"\n👤 Human chose: {label_to_letter(row['LABEL'])}")
print(f"🤖 LLM chose: {mismatches.iloc[0]['LLM']}")


import openai

def judge_with_gpt_arena(prompt, response_a, response_b):
    client = openai.OpenAI(api_key="sk-...")  # ← Replace with your real key

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are acting as a human judge in the Chatbot Arena preference dataset. "
                    "You are shown a user prompt and two AI-generated responses. Your task is to evaluate which response a human user "
                    "would find more satisfactory overall. You must consider helpfulness, factual accuracy, clarity, and alignment with the prompt. "
                    "You must avoid favoring responses due to their position, verbosity, or promotional tone. "
                    "After comparing the responses, reply with only one word: A, B, or Tie."
                ),
            },
            {
                "role": "user",
                "content": f"""Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response would a human prefer? Reply with one word only: A, B, or Tie.""",
            },
        ],
        temperature=0,
    )

    choice = response.choices[0].message.content.strip().upper()

    if choice.startswith("A"):
        return "A"
    elif choice.startswith("B"):
        return "B"
    elif choice.startswith("T"):
        return "Tie"
    else:
        return "?"


    client = openai.OpenAI(api_key="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")  # Replace with your actual API key

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are acting as a human judge in the Chatbot Arena preference dataset. "
                    "You are shown a user prompt and two AI-generated responses. Your task is to evaluate which response a human user "
                    "would find more satisfactory overall. You must consider helpfulness, factual accuracy, clarity, and alignment with the prompt. "
                    "You must avoid favoring responses due to their position, verbosity, or promotional tone. "
                    "After comparing the responses, reply with only one word: A, B, or Tie."
                ),
            },
            {
                "role": "user",
                "content": f"""Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response would a human prefer? Reply with one word only: A, B, or Tie.""",
            },
        ],
        temperature=0,
    )

    choice = response.choices[0].message.content.strip().upper()

    if choice.startswith("A"):
        return "A"
    elif choice.startswith("B"):
        return "B"
    elif choice.startswith("T"):
        return "Tie"
    else:
        return "?"


import openai

def judge_with_gpt_arena(prompt, response_a, response_b):
    client = openai.OpenAI(api_key="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")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are acting as a human judge in the Chatbot Arena preference dataset. "
                    "You are shown a user prompt and two AI-generated responses. Your task is to evaluate which response a human user "
                    "would find more satisfactory overall. You must consider helpfulness, factual accuracy, clarity, and alignment with the prompt. "
                    "You must avoid favoring responses due to their position, verbosity, or promotional tone. "
                    "After comparing the responses, reply with only one word: A, B, or Tie."
                ),
            },
            {
                "role": "user",
                "content": f"""Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response would a human prefer? Reply with one word only: A, B, or Tie.""",
            },
        ],
        temperature=0,
    )

    choice = response.choices[0].message.content.strip().upper()

    if choice.startswith("A"):
        return "A"
    elif choice.startswith("B"):
        return "B"
    elif choice.startswith("T"):
        return "Tie"
    else:
        return "?"


import time

results = []

for i, row in val_df.head(10).iterrows():
    try:
        print(f"⏳ Row {i+1}/10...", end=' ', flush=True)
        llm = judge_with_gpt_arena(row['prompt'], row['response_a'], row['response_b'])
        human = label_to_letter(row['LABEL'])
        match = llm == human
        results.append({"Row": i, "LLM": llm, "Human": human, "Match": match})
        print(f"✅ ({llm} vs {human})")
        time.sleep(1.5)  # Sleep to avoid rate limit
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({
            "Row": i,
            "LLM": "ERROR",
            "Human": label_to_letter(row['LABEL']),
            "Match": False,
            "Error": str(e)
        })


results_df = pd.DataFrame(results)
results_df
accuracy = results_df['Match'].mean()
print(f"🎯 LLM-Human Agreement: {accuracy:.2%}")


import openai
import pandas as pd

openai.api_key = "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"

def evaluate_heuristics(index, prompt, response_a, response_b, label):
    system_msg = {
        "role": "system",
        "content": (
            "You are a heuristic evaluator tasked with modeling how a human would judge two AI responses. "
            "You will evaluate the following heuristics:\n"
            "1. Is the prompt fact-seeking?\n"
            "2. Does the selected response provide direct factual information?\n"
            "3. Does it include the keyword or concept from the prompt early in the response?\n"
            "4. Does it avoid hedging or disclaimers like 'As an AI language model...'\n\n"
            "Return one line: either 'YES - heuristically justified' or 'NO - heuristically unjustified'."
        )
    }

    selected = response_a if label == 0 else response_b if label == 1 else "Tie"

    if selected == "Tie":
        return {"Index": index, "Heuristic": "TIE - Skipped"}

    user_msg = {
        "role": "user",
        "content": f"""Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

The human selected {'Response A' if label == 0 else 'Response B'}.

Evaluate the selected response using the four heuristics described. Do not explain. Just return your judgment:
"""
    }

    try:
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[system_msg, user_msg],
            temperature=0
        )
        verdict = result.choices[0].message.content.strip()
        return {"Index": index, "Heuristic": verdict}
    except Exception as e:
        return {"Index": index, "Heuristic": f"ERROR: {str(e)}"}

# Run the test
heuristic_results = []
for _, row in df.head(10).iterrows():
    heuristic_results.append(evaluate_heuristics(
        index=row["Column1"],
        prompt=row["prompt"],
        response_a=row["response_a"],
        response_b=row["response_b"],
        label=row["LABEL"]
    ))

# Display the table
pd.DataFrame(heuristic_results)


import openai
import pandas as pd
import time

# Setup client
client = openai.OpenAI(api_key="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")

# Heuristic evaluation function
def evaluate_heuristics(index, prompt, response_a, response_b, label):
    system_msg = {
        "role": "system",
        "content": (
            "You are a heuristic evaluator tasked with modeling how a human would judge two AI responses. "
            "You will evaluate the following heuristics:\n"
            "1. Is the prompt fact-seeking?\n"
            "2. Does the selected response provide direct factual information?\n"
            "3. Does it include the keyword or concept from the prompt early in the response?\n"
            "4. Does it avoid hedging or disclaimers like 'As an AI language model...'\n\n"
            "Return one line: either 'YES - heuristically justified' or 'NO - heuristically unjustified'."
        )
    }

    selected = response_a if label == 0 else response_b if label == 1 else "Tie"

    if selected == "Tie":
        return {"Index": index, "Heuristic": "TIE - Skipped"}

    user_msg = {
        "role": "user",
        "content": f"""Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

The human selected {'Response A' if label == 0 else 'Response B'}.

Evaluate the selected response using the four heuristics described. Do not explain. Just return your judgment:"""
    }

    try:
        result = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_msg, user_msg],
            temperature=0
        )
        verdict = result.choices[0].message.content.strip()
        return {"Index": index, "Heuristic": verdict}
    except Exception as e:
        return {"Index": index, "Heuristic": f"ERROR: {str(e)}"}

# Run on first 10 rows
heuristic_results = []
for _, row in df.head(10).iterrows():
    print(f"Evaluating row {row['Column1']}...", end=' ')
    result = evaluate_heuristics(
        index=row["Column1"],
        prompt=row["prompt"],
        response_a=row["response_a"],
        response_b=row["response_b"],
        label=row["LABEL"]
    )
    print(result["Heuristic"])
    heuristic_results.append(result)
    time.sleep(1.5)

# Display final DataFrame
pd.DataFrame(heuristic_results)


import openai
import pandas as pd
import time

client = openai.OpenAI(api_key="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")

def get_factuality_flags(index, prompt, response_a, response_b):
    system_msg = {
        "role": "system",
        "content": (
            "You are a binary classifier. Given a prompt and two responses, your job is to decide:\n"
            "- Is the prompt fact-seeking? (1 = yes, 0 = no)\n"
            "- Does response A provide direct factual information? (1/0)\n"
            "- Does response B provide direct factual information? (1/0)\n\n"
            "Return the result as a list of 3 binary values in the format: [f_q, f_a, f_b]. Do not explain anything."
        )
    }

    user_msg = {
        "role": "user",
        "content": f"""Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Return just a list of 3 binary values: [f_q, f_a, f_b]
"""
    }

    try:
        result = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_msg, user_msg],
            temperature=0
        )
        flags = result.choices[0].message.content.strip()
        return {"Index": index, "Factuality": flags}
    except Exception as e:
        return {"Index": index, "Factuality": f"ERROR: {str(e)}"}

# Run on first 10 rows
factuality_results = []
for _, row in df.head(10).iterrows():
    print(f"Row {row['Column1']}...", end=' ')
    result = get_factuality_flags(
        index=row["Column1"],
        prompt=row["prompt"],
        response_a=row["response_a"],
        response_b=row["response_b"]
    )
    print(result["Factuality"])
    factuality_results.append(result)
    time.sleep(1.5)

# Show results
pd.DataFrame(factuality_results)


import ast
import pandas as pd

# Assume factuality_results already exists
# Parse [f_q, f_a, f_b] into separate fields
parsed_rows = []
for row in factuality_results:
    try:
        flags = ast.literal_eval(row["Factuality"])
        if isinstance(flags, list) and len(flags) == 3:
            f_q, f_a, f_b = flags
            parsed_rows.append({
                "Index": row["Index"],
                "f_q": int(f_q),
                "f_a": int(f_a),
                "f_b": int(f_b)
            })
    except:
        continue

factual_df = pd.DataFrame(parsed_rows)

# Merge factual flags with true LABELs
merged_df = pd.merge(factual_df, df[["Column1", "LABEL"]], left_on="Index", right_on="Column1", how="left")

# Define prediction rule
def check_alignment(row):
    if row["f_q"] != 1:
        return "Not fact-seeking"
    if row["f_a"] == 1 and row["f_b"] == 0:
        return "Correct" if row["LABEL"] == 0 else "Mismatch"
    if row["f_a"] == 0 and row["f_b"] == 1:
        return "Correct" if row["LABEL"] == 1 else "Mismatch"
    if row["f_a"] == 1 and row["f_b"] == 1:
        return "Both factual"
    if row["f_a"] == 0 and row["f_b"] == 0:
        return "Neither factual"
    return "Other"

merged_df["Alignment"] = merged_df.apply(check_alignment, axis=1)

# Show results
merged_df[["Index", "f_q", "f_a", "f_b", "LABEL", "Alignment"]]


def evaluate_truth_alignment(row):
    fq, fa, fb, label = row["f_q"], row["f_a"], row["f_b"], row["LABEL"]

    if fq == 1 and fa == 1 and fb == 0 and label == 0:
        return "GOOD"
    elif fq == 1 and fa == 0 and fb == 1 and label == 1:
        return "GOOD"
    elif fq == 1 and fa == 1 and fb == 1 and label == 2:
        return "GOOD"
    elif fq == 0 and fa == 1 and fb == 1 and label == 2:
        return "GOOD"
    elif fq == 0 and fa == 0 and fb == 0 and label == 2:
        return "GOOD"
    else:
        return "BAD"

# Apply the rule to each row
merged_df["Truth Alignment"] = merged_df.apply(evaluate_truth_alignment, axis=1)

# Show results
merged_df[["Index", "f_q", "f_a", "f_b", "LABEL", "Truth Alignment"]]


import random
import ast
import pandas as pd

# Reuse your rule
def evaluate_truth_alignment(f_q, f_a, f_b, label):
    if f_q == 1 and f_a == 1 and f_b == 0 and label == 0:
        return "GOOD"
    elif f_q == 1 and f_a == 0 and f_b == 1 and label == 1:
        return "GOOD"
    elif f_q == 1 and f_a == 1 and f_b == 1 and label == 2:
        return "GOOD"
    elif f_q == 0 and f_a == 1 and f_b == 1 and label == 2:
        return "GOOD"
    elif f_q == 0 and f_a == 0 and f_b == 0 and label == 2:
        return "GOOD"
    else:
        return "BAD"

# Store final results
trial_results = []

# Run 10 trials
for trial_num in range(1, 11):
    sample = df.sample(n=10, random_state=random.randint(1, 10000))
    good_count = 0
    total = 0

    for _, row in sample.iterrows():
        try:
            flags = ast.literal_eval(get_factuality_flags(
                index=row["Column1"],
                prompt=row["prompt"],
                response_a=row["response_a"],
                response_b=row["response_b"]
            )["Factuality"])

            if isinstance(flags, list) and len(flags) == 3:
                f_q, f_a, f_b = map(int, flags)
                label = row["LABEL"]
                result = evaluate_truth_alignment(f_q, f_a, f_b, label)
                total += 1
                if result == "GOOD":
                    good_count += 1

        except Exception as e:
            continue

    percent_good = (good_count / total) * 100 if total > 0 else 0
    trial_results.append({
        "Trial": trial_num,
        "GOOD Count": good_count,
        "Total Evaluated": total,
        "Percent GOOD": round(percent_good, 2)
    })

# Display summary table
summary_df = pd.DataFrame(trial_results)
summary_df


import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def evaluate_keyword_salience(prompt, response):
    doc = nlp(prompt)
    
    # Extract all noun lemmas from the prompt
    prompt_nouns = set([token.lemma_.lower() for token in doc if token.pos_ == "NOUN"])

    if not prompt_nouns:
        return 0  # No nouns to compare

    # Only look at the first half of the response
    halfway = len(response) // 2
    response_section = response[:halfway].lower()

    # Check if any noun is present in first half
    for noun in prompt_nouns:
        if noun in response_section:
            return 1  # Match found

    return 0  # No match


def keyword_salience_winner(prompt, response_a, response_b):
    score_a = evaluate_keyword_salience(prompt, response_a)
    score_b = evaluate_keyword_salience(prompt, response_b)

    if score_a > score_b:
        return 0  # A wins
    elif score_b > score_a:
        return 1  # B wins
    else:
        return 2  # Tie


import pandas as pd
import random

keyword_trial_results = []

for trial_num in range(1, 11):
    sample = df.sample(n=10, random_state=random.randint(1, 10000))
    good_count = 0
    total = 0

    for _, row in sample.iterrows():
        try:
            predicted = keyword_salience_winner(
                prompt=row["prompt"],
                response_a=row["response_a"],
                response_b=row["response_b"]
            )
            if predicted == row["LABEL"]:
                good_count += 1
            total += 1
        except Exception as e:
            continue

    percent_good = (good_count / total) * 100 if total > 0 else 0
    keyword_trial_results.append({
        "Trial": trial_num,
        "GOOD Count": good_count,
        "Total Evaluated": total,
        "Percent GOOD": round(percent_good, 2)
    })

# Display summary table
keyword_summary_df = pd.DataFrame(keyword_trial_results)
keyword_summary_df


def red_flag_all_caps(response):
    words = response.split()
    if not words:
        return 0
    cap_words = [w for w in words if w.isupper() and len(w) > 3]
    return 1 if len(cap_words) > 0 and (len(cap_words) / len(words)) > 0.5 else 0


def red_flag_refusal(response):
    text = response.lower()
    refusal_phrases = [
        "i cannot", "i'm unable", "as an ai", "i don't have opinions",
        "i do not have personal beliefs", "i can't help with that", "i cannot provide", 
        "i am not qualified", "i am not able", "i am unable", "my training data",
        "i don't have access to", "i don't possess", "i cannot make judgments",
        "i am not allowed", "i was not trained to"
    ]
    return 1 if any(phrase in text for phrase in refusal_phrases) else 0



# === WEIGHTS ===
W_KEYWORD = 10
W_FACTUALITY = 20
W_ALL_CAPS = -50
W_REFUSAL = -50

# Compute MVAT score for one response (A or B)
def compute_mvat_score(prompt, response, is_a, factuality_flags):
    score = 0

    # 1. Keyword salience
    if evaluate_keyword_salience(prompt, response) == 1:
        score += W_KEYWORD

    # 2. Factuality (only if f_q == 1)
    f_q, f_a, f_b = factuality_flags
    if f_q == 1:
        if is_a and f_a == 1:
            score += W_FACTUALITY
        elif not is_a and f_b == 1:
            score += W_FACTUALITY

    # 3. Red flags
    if red_flag_all_caps(response):
        score += W_ALL_CAPS
    if red_flag_refusal(response):
        score += W_REFUSAL

    return score


from sentence_transformers import SentenceTransformer, util

# Load local embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
def compute_originality_scores(prompt, response_a, response_b):
    embeddings = embedder.encode([prompt, response_a, response_b], convert_to_tensor=True)

    sim_pa = util.cos_sim(embeddings[0], embeddings[1]).item()
    sim_pb = util.cos_sim(embeddings[0], embeddings[2]).item()
    sim_ab = util.cos_sim(embeddings[1], embeddings[2]).item()

    orig_a = sim_pa - sim_ab
    orig_b = sim_pb - sim_ab

    return {"orig_a": orig_a, "orig_b": orig_b}




def compute_cosine_scores(prompt, response_a, response_b):
    embeddings = embedder.encode([prompt, response_a, response_b], convert_to_tensor=True)

    sim_pa = util.cos_sim(embeddings[0], embeddings[1]).item()
    sim_pb = util.cos_sim(embeddings[0], embeddings[2]).item()

    return {"sim_pa": sim_pa, "sim_pb": sim_pb}


def compute_cosine_scores(prompt, response_a, response_b):
    embeddings = embedder.encode([prompt, response_a, response_b], convert_to_tensor=True)

    sim_pa = util.cos_sim(embeddings[0], embeddings[1]).item()
    sim_pb = util.cos_sim(embeddings[0], embeddings[2]).item()

    return {"sim_pa": sim_pa, "sim_pb": sim_pb}


# === WEIGHTS ===
W_KEYWORD = 10
W_FACTUALITY = 20
W_ALL_CAPS = -50
W_REFUSAL = -50
W_TOO_SHORT = -5
W_OVER_PUNCT = -3
W_HIGH_VERBOSITY = -2
W_ORIGINALITY = 5
W_COSINE = 5

import re

def compute_mvat_score(prompt, response, is_a, factuality_flags, originality_scores, cosine_scores):
    score = 0

    # 1. Keyword salience
    if evaluate_keyword_salience(prompt, response) == 1:
        score += W_KEYWORD

    # 2. Factuality
    f_q, f_a, f_b = factuality_flags
    if f_q == 1:
        if is_a and f_a == 1:
            score += W_FACTUALITY
        elif not is_a and f_b == 1:
            score += W_FACTUALITY

    # 3. Red flags
    if red_flag_all_caps(response):
        score += W_ALL_CAPS
    if red_flag_refusal(response):
        score += W_REFUSAL

    # 4. Syntactic scoring
    length = len(response)
    word_count = len(response.split())
    punct_count = len(re.findall(r'[^\w\s]', response))
    verbosity_ratio = word_count / length if length > 0 else 0

    if length < 50:
        score += W_TOO_SHORT
    if punct_count > 25:
        score += W_OVER_PUNCT
    if verbosity_ratio > 0.2:
        score += W_HIGH_VERBOSITY

    # 5. Originality
    if originality_scores:
        if is_a:
            score += W_ORIGINALITY * originality_scores["orig_a"]
        else:
            score += W_ORIGINALITY * originality_scores["orig_b"]

    # 6. Cosine similarity to prompt
    if cosine_scores:
        if is_a:
            score += W_COSINE * cosine_scores["sim_pa"]
        else:
            score += W_COSINE * cosine_scores["sim_pb"]

    return score


import random
import ast

mvat_results = []

# Sample 10 random rows
sample = df.sample(n=10, random_state=random.randint(1, 10000))

for _, row in sample.iterrows():
    try:
        # Get factuality
        flags = ast.literal_eval(get_factuality_flags(
            index=row["Column1"],
            prompt=row["prompt"],
            response_a=row["response_a"],
            response_b=row["response_b"]
        )["Factuality"])
        f_q, f_a, f_b = map(int, flags)

        # Get embeddings
        originality_scores = compute_originality_scores(row["prompt"], row["response_a"], row["response_b"])
        cosine_scores = compute_cosine_scores(row["prompt"], row["response_a"], row["response_b"])

        # Score each response
        score_a = compute_mvat_score(
            prompt=row["prompt"],
            response=row["response_a"],
            is_a=True,
            factuality_flags=(f_q, f_a, f_b),
            originality_scores=originality_scores,
            cosine_scores=cosine_scores
        )

        score_b = compute_mvat_score(
            prompt=row["prompt"],
            response=row["response_b"],
            is_a=False,
            factuality_flags=(f_q, f_a, f_b),
            originality_scores=originality_scores,
            cosine_scores=cosine_scores
        )

        predicted = 0 if score_a > score_b else 1 if score_b > score_a else 2
        actual = row["LABEL"]

        mvat_results.append({
            "Index": row["Column1"],
            "Score A": round(score_a, 2),
            "Score B": round(score_b, 2),
            "Predicted Label": predicted,
            "Actual Label": actual,
            "Match": predicted == actual
        })

    except Exception as e:
        mvat_results.append({
            "Index": row["Column1"],
            "Score A": None,
            "Score B": None,
            "Predicted Label": "ERROR",
            "Actual Label": row["LABEL"],
            "Match": False,
            "Error": str(e)
        })

# Convert to DataFrame
mvat_df = pd.DataFrame(mvat_results)
mvat_df


import openai

client = openai.OpenAI(api_key="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")  # Replace with your actual key

def get_factuality_flags(index, prompt, response_a, response_b):
    system_msg = {
        "role": "system",
        "content": (
            "You are a binary classifier. Given a prompt and two responses, your job is to decide:\n"
            "- Is the prompt fact-seeking? (1 = yes, 0 = no)\n"
            "- Does response A provide direct factual information? (1/0)\n"
            "- Does response B provide direct factual information? (1/0)\n\n"
            "Return the result as a list of 3 binary values in the format: [f_q, f_a, f_b]. Do not explain anything."
        )
    }

    user_msg = {
        "role": "user",
        "content": f"""Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Return just a list of 3 binary values: [f_q, f_a, f_b]
"""
    }

    try:
        result = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_msg, user_msg],
            temperature=0
        )
        flags = result.choices[0].message.content.strip()
        return {"Index": index, "Factuality": flags}
    except Exception as e:
        return {"Index": index, "Factuality": f"[0, 0, 0]"}  # Fail-safe default


import openai

client = openai.OpenAI(api_key="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")

def get_factuality_flags(index, prompt, response_a, response_b):
    system_msg = {
        "role": "system",
        "content": (
            "You are a binary classifier. Given a prompt and two responses, your job is to decide:\n"
            "- Is the prompt fact-seeking? (1 = yes, 0 = no)\n"
            "- Does response A provide direct factual information? (1/0)\n"
            "- Does response B provide direct factual information? (1/0)\n\n"
            "Return the result as a list of 3 binary values in the format: [f_q, f_a, f_b]. Do not explain anything."
        )
    }

    user_msg = {
        "role": "user",
        "content": f"""Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Return just a list of 3 binary values: [f_q, f_a, f_b]
"""
    }

    try:
        result = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_msg, user_msg],
            temperature=0
        )
        flags = result.choices[0].message.content.strip()
        return {"Index": index, "Factuality": flags}
    except Exception:
        return {"Index": index, "Factuality": "[0, 0, 0]"}  # Fallback default


import openai

client = openai.OpenAI(api_key="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")

def get_factuality_flags(index, prompt, response_a, response_b):
    system_msg = {
        "role": "system",
        "content": (
            "You are a binary classifier. Given a prompt and two responses, your job is to decide:\n"
            "- Is the prompt fact-seeking? (1 = yes, 0 = no)\n"
            "- Does response A provide direct factual information? (1/0)\n"
            "- Does response B provide direct factual information? (1/0)\n\n"
            "Return the result as a list of 3 binary values in the format: [f_q, f_a, f_b]. Do not explain anything."
        )
    }

    user_msg = {
        "role": "user",
        "content": f"""Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Return just a list of 3 binary values: [f_q, f_a, f_b]
"""
    }

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_msg, user_msg],
            temperature=0
        )
        flags = response.choices[0].message.content.strip()
        print(f"[{index}] FLAGS: {flags}")  # Debug output
        return {"Index": index, "Factuality": flags}
    except Exception as e:
        print(f"[{index}] ERROR: {str(e)}")  # Debug output
        return {"Index": index, "Factuality": "[0, 0, 0]"}


get_factuality_flags(
    index="TEST",
    prompt="What is the capital of France?",
    response_a="The capital of France is Paris.",
    response_b="I don't know the capital of France."
)


import random
import ast

mvat_trial_results = []

for trial_num in range(1, 11):
    matches = 0
    total = 0
    print(f"\n=== Trial {trial_num} ===")

    sample = df.sample(n=10, random_state=random.randint(1, 10000))

    for _, row in sample.iterrows():
        try:
            # print(f"⏳ Running on row: {row['Column1']}")

            # Get factuality
            flags_result = get_factuality_flags(
                index=row["Column1"],
                prompt=row["prompt"],
                response_a=row["response_a"],
                response_b=row["response_b"]
            )
            flags = ast.literal_eval(flags_result["Factuality"])
            f_q, f_a, f_b = map(int, flags)

            # Get embeddings
            originality_scores = compute_originality_scores(
                row["prompt"], row["response_a"], row["response_b"]
            )
            cosine_scores = compute_cosine_scores(
                row["prompt"], row["response_a"], row["response_b"]
            )

            # MVAT scores
            score_a = compute_mvat_score(
                prompt=row["prompt"],
                response=row["response_a"],
                is_a=True,
                factuality_flags=(f_q, f_a, f_b),
                originality_scores=originality_scores,
                cosine_scores=cosine_scores
            )
            score_b = compute_mvat_score(
                prompt=row["prompt"],
                response=row["response_b"],
                is_a=False,
                factuality_flags=(f_q, f_a, f_b),
                originality_scores=originality_scores,
                cosine_scores=cosine_scores
            )

            predicted = 0 if score_a > score_b else 1 if score_b > score_a else 2
            actual = row["LABEL"]

            if predicted == actual:
                matches += 1
            total += 1
            # print(f"✅ Row {row['Column1']}: Predicted={predicted}, Actual={actual}")

        except Exception as e:
            print(f"❌ ERROR on row {row['Column1']}: {str(e)}")
            continue

    accuracy = (matches / total) * 100 if total > 0 else 0
    mvat_trial_results.append({
        "Trial": trial_num,
        "Correct Predictions": matches,
        "Total Evaluated": total,
        "Accuracy (%)": round(accuracy, 2)
    })

# Summary DataFrame
mvat_summary_df = pd.DataFrame(mvat_trial_results)
mvat_summary_df


import random
import ast

mvat_trial_results = []

for trial_num in range(1, 11):
    matches = 0
    total = 0
    sample = df.sample(n=10, random_state=random.randint(1, 10000))

    for _, row in sample.iterrows():
        try:
            flags_result = get_factuality_flags(
                index=row["Column1"],
                prompt=row["prompt"],
                response_a=row["response_a"],
                response_b=row["response_b"]
            )
            flags = ast.literal_eval(flags_result["Factuality"])
            f_q, f_a, f_b = map(int, flags)

            originality_scores = compute_originality_scores(
                row["prompt"], row["response_a"], row["response_b"]
            )
            cosine_scores = compute_cosine_scores(
                row["prompt"], row["response_a"], row["response_b"]
            )

            score_a = compute_mvat_score(
                prompt=row["prompt"],
                response=row["response_a"],
                is_a=True,
                factuality_flags=(f_q, f_a, f_b),
                originality_scores=originality_scores,
                cosine_scores=cosine_scores
            )
            score_b = compute_mvat_score(
                prompt=row["prompt"],
                response=row["response_b"],
                is_a=False,
                factuality_flags=(f_q, f_a, f_b),
                originality_scores=originality_scores,
                cosine_scores=cosine_scores
            )

            predicted = 0 if score_a > score_b else 1 if score_b > score_a else 2
            actual = row["LABEL"]

            if predicted == actual:
                matches += 1
            total += 1

        except:
            continue

    accuracy = (matches / total) * 100 if total > 0 else 0
    mvat_trial_results.append({
        "Trial": trial_num,
        "Correct Predictions": matches,
        "Total Evaluated": total,
        "Accuracy (%)": round(accuracy, 2)
    })

mvat_summary_df = pd.DataFrame(mvat_trial_results)
import IPython.display as disp
disp.display(mvat_summary_df)


import random
import pandas as pd
import ast
import IPython.display as disp

# === Main Evaluation Function: MVAT Batch Tester ===
def run_mvat_trials(df, num_trials=10, rows_per_trial=10):
    trial_results = []

    for trial_num in range(1, num_trials + 1):
        sample = df.sample(n=rows_per_trial, random_state=random.randint(1, 10000))
        correct = 0
        total = 0

        for _, row in sample.iterrows():
            try:
                # Get factuality flags
                flags = ast.literal_eval(get_factuality_flags(
                    index=row["Column1"],
                    prompt=row["prompt"],
                    response_a=row["response_a"],
                    response_b=row["response_b"]
                )["Factuality"])

                # Compute scores
                score_a = compute_mvat_score(row["prompt"], row["response_a"], is_a=True, factuality_flags=flags)
                score_b = compute_mvat_score(row["prompt"], row["response_b"], is_a=False, factuality_flags=flags)

                predicted = 0 if score_a > score_b else 1 if score_b > score_a else 2
                actual = row["LABEL"]
                match = predicted == actual
                if match:
                    correct += 1
                total += 1

            except:
                continue

        percent_correct = (correct / total) * 100 if total > 0 else 0
        trial_results.append({
            "Trial": trial_num,
            "Correct Predictions": correct,
            "Total Evaluated": total,
            "Accuracy (%)": round(percent_correct, 2)
        })

    mvat_summary_df = pd.DataFrame(trial_results)
    disp.display(mvat_summary_df)



run_mvat_trials(df)

row = df.sample(1).iloc[0]

try:
    print("🔍 Testing factuality flags on 1 row...")
    flags = ast.literal_eval(get_factuality_flags(
        index=row["Column1"],
        prompt=row["prompt"],
        response_a=row["response_a"],
        response_b=row["response_b"]
    )["Factuality"])
    print("✅ FLAGS:", flags)
except Exception as e:
    print("❌ ERROR:", e)



import random
import pandas as pd
import ast

def run_mvat_trials_final(df, num_trials=10, rows_per_trial=10):
    trial_results = []

    for trial_num in range(1, num_trials + 1):
        correct = 0
        total = 0
        sample = df.sample(n=rows_per_trial, random_state=random.randint(1, 10000))

        for _, row in sample.iterrows():
            try:
                # Get factuality
                flags = ast.literal_eval(get_factuality_flags(
                    index=row["Column1"],
                    prompt=row["prompt"],
                    response_a=row["response_a"],
                    response_b=row["response_b"]
                )["Factuality"])
                if not (isinstance(flags, list) and len(flags) == 3):
                    continue
                f_q, f_a, f_b = map(int, flags)

                # Get embeddings
                originality_scores = compute_originality_scores(
                    row["prompt"], row["response_a"], row["response_b"]
                )
                cosine_scores = compute_cosine_scores(
                    row["prompt"], row["response_a"], row["response_b"]
                )

                # Score
                score_a = compute_mvat_score(
                    row["prompt"], row["response_a"], True,
                    factuality_flags=(f_q, f_a, f_b),
                    originality_scores=originality_scores,
                    cosine_scores=cosine_scores
                )
                score_b = compute_mvat_score(
                    row["prompt"], row["response_b"], False,
                    factuality_flags=(f_q, f_a, f_b),
                    originality_scores=originality_scores,
                    cosine_scores=cosine_scores
                )

                predicted = 0 if score_a > score_b else 1 if score_b > score_a else 2
                if predicted == row["LABEL"]:
                    correct += 1
                total += 1

            except Exception:
                continue

        accuracy = (correct / total) * 100 if total > 0 else 0
        trial_results.append({
            "Trial": trial_num,
            "Correct Predictions": correct,
            "Total Evaluated": total,
            "Accuracy (%)": round(accuracy, 2)
        })

    return pd.DataFrame(trial_results)


mvat_results = run_mvat_trials_final(df)
mvat_results

import openai

client = openai.OpenAI(api_key="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")

def get_factuality_flags(index, prompt, response_a, response_b):
    system_msg = {
        "role": "system",
        "content": (
            "You are a binary classifier. Given a prompt and two responses, your job is to decide:\n"
            "- Is the prompt fact-seeking? (1 = yes, 0 = no)\n"
            "- Does response A provide direct factual information? (1/0)\n"
            "- Does response B provide direct factual information? (1/0)\n\n"
            "Return the result as a list of 3 binary values in the format: [f_q, f_a, f_b]. Do not explain anything."
        )
    }

    user_msg = {
        "role": "user",
        "content": f"""Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Return just a list of 3 binary values: [f_q, f_a, f_b]
"""
    }

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_msg, user_msg],
            temperature=0
        )
        flags = response.choices[0].message.content.strip()
        return {"Index": index, "Factuality": flags}
    except:
        return {"Index": index, "Factuality": "[0, 0, 0]"}


mvat_results = run_mvat_trials_final_clean(df)
mvat_results




import random
import pandas as pd
import ast

def run_mvat_trials_final_clean(df, num_trials=10, rows_per_trial=10):
    results = []

    for trial_num in range(1, num_trials + 1):
        correct = 0
        total = 0
        sample = df.sample(n=rows_per_trial, random_state=random.randint(1, 10000))

        for _, row in sample.iterrows():
            try:
                flags = ast.literal_eval(get_factuality_flags(
                    index=row["Column1"],
                    prompt=row["prompt"],
                    response_a=row["response_a"],
                    response_b=row["response_b"]
                )["Factuality"])

                if not (isinstance(flags, list) and len(flags) == 3):
                    continue

                f_q, f_a, f_b = map(int, flags)

                originality_scores = compute_originality_scores(
                    row["prompt"], row["response_a"], row["response_b"]
                )
                cosine_scores = compute_cosine_scores(
                    row["prompt"], row["response_a"], row["response_b"]
                )

                score_a = compute_mvat_score(
                    row["prompt"], row["response_a"], True,
                    factuality_flags=(f_q, f_a, f_b),
                    originality_scores=originality_scores,
                    cosine_scores=cosine_scores
                )
                score_b = compute_mvat_score(
                    row["prompt"], row["response_b"], False,
                    factuality_flags=(f_q, f_a, f_b),
                    originality_scores=originality_scores,
                    cosine_scores=cosine_scores
                )

                predicted = 0 if score_a > score_b else 1 if score_b > score_a else 2
                actual = row["LABEL"]

                if predicted == actual:
                    correct += 1
                total += 1

            except:
                continue

        accuracy = (correct / total) * 100 if total > 0 else 0
        results.append({
            "Trial": trial_num,
            "Correct Predictions": correct,
            "Total Evaluated": total,
            "Accuracy (%)": round(accuracy, 2)
        })

    return pd.DataFrame(results)




mvat_results = run_mvat_trials_final_clean(df)
mvat_results

import random
import pandas as pd
import ast

def extract_mvat_training_data(df, sample_size=500):
    training_rows = []

    sample = df.sample(n=sample_size, random_state=random.randint(1, 10000))

    for _, row in sample.iterrows():
        try:
            # === Factuality
            flags = ast.literal_eval(get_factuality_flags(
                index=row["Column1"],
                prompt=row["prompt"],
                response_a=row["response_a"],
                response_b=row["response_b"]
            )["Factuality"])
            if not (isinstance(flags, list) and len(flags) == 3):
                continue
            f_q, f_a, f_b = map(int, flags)

            # === Red Flags
            refusal_a = red_flag_refusal(row["response_a"])
            refusal_b = red_flag_refusal(row["response_b"])
            caps_a = red_flag_all_caps(row["response_a"])
            caps_b = red_flag_all_caps(row["response_b"])

            # === Keyword
            keyword_a = evaluate_keyword_salience(row["prompt"], row["response_a"])
            keyword_b = evaluate_keyword_salience(row["prompt"], row["response_b"])

            # === Embedding features
            originality = compute_originality_scores(row["prompt"], row["response_a"], row["response_b"])
            cosine = compute_cosine_scores(row["prompt"], row["response_a"], row["response_b"])

            # === Append full feature set
            training_rows.append({
                "Index": row["Column1"],
                "f_q": f_q,
                "f_a": f_a,
                "f_b": f_b,
                "keyword_a": keyword_a,
                "keyword_b": keyword_b,
                "refusal_a": refusal_a,
                "refusal_b": refusal_b,
                "caps_a": caps_a,
                "caps_b": caps_b,
                "orig_a": originality["orig_a"],
                "orig_b": originality["orig_b"],
                "cosine_a": cosine["sim_pa"],
                "cosine_b": cosine["sim_pb"],
                "LABEL": row["LABEL"]
            })

        except Exception:
            continue

    return pd.DataFrame(training_rows)

# === Run the function and return the table
training_df = extract_mvat_training_data(df, sample_size=500)
training_df



training_df.to_excel("mvat_training_data.xlsx", index=False)


import random
import pandas as pd
import ast
from sentence_transformers import util

def extract_mvat_training_data_batched(df, sample_size=50, batch_size=10):
    rows = []
    sample = df.sample(n=sample_size, random_state=random.randint(1, 10000))

    prompts = []
    responses_a = []
    responses_b = []
    metadata = []

    for _, row in sample.iterrows():
        try:
            # === Get factuality
            flags = ast.literal_eval(get_factuality_flags(
                index=row["Column1"],
                prompt=row["prompt"],
                response_a=row["response_a"],
                response_b=row["response_b"]
            )["Factuality"])
            if not (isinstance(flags, list) and len(flags) == 3):
                continue
            f_q, f_a, f_b = map(int, flags)

            # === Syntactic + Heuristic Features
            refusal_a = red_flag_refusal(row["response_a"])
            refusal_b = red_flag_refusal(row["response_b"])
            caps_a = red_flag_all_caps(row["response_a"])
            caps_b = red_flag_all_caps(row["response_b"])
            keyword_a = evaluate_keyword_salience(row["prompt"], row["response_a"])
            keyword_b = evaluate_keyword_salience(row["prompt"], row["response_b"])

            prompts.append(row["prompt"])
            responses_a.append(row["response_a"])
            responses_b.append(row["response_b"])
            metadata.append({
                "Index": row["Column1"],
                "f_q": f_q,
                "f_a": f_a,
                "f_b": f_b,
                "keyword_a": keyword_a,
                "keyword_b": keyword_b,
                "refusal_a": refusal_a,
                "refusal_b": refusal_b,
                "caps_a": caps_a,
                "caps_b": caps_b,
                "LABEL": row["LABEL"]
            })

        except Exception:
            continue

    # === Batch embeddings using embedder
    orig_a_list = []
    orig_b_list = []

    for i in range(0, len(prompts), batch_size):
        p_batch = prompts[i:i + batch_size]
        a_batch = responses_a[i:i + batch_size]
        b_batch = responses_b[i:i + batch_size]

        emb_p = embedder.encode(p_batch, convert_to_tensor=True)
        emb_a = embedder.encode(a_batch, convert_to_tensor=True)
        emb_b = embedder.encode(b_batch, convert_to_tensor=True)

        for j in range(len(p_batch)):
            sim_pa = util.cos_sim(emb_p[j], emb_a[j]).item()
            sim_pb = util.cos_sim(emb_p[j], emb_b[j]).item()
            sim_ab = util.cos_sim(emb_a[j], emb_b[j]).item()

            orig_a_list.append(sim_pa - sim_ab)
            orig_b_list.append(sim_pb - sim_ab)

    # === Final row assembly
    for i, meta in enumerate(metadata):
        row = meta.copy()
        row["orig_a"] = orig_a_list[i]
        row["orig_b"] = orig_b_list[i]
        rows.append(row)

    return pd.DataFrame(rows)


training_df = extract_mvat_training_data_batched(df, sample_size=50)
training_df.to_excel("mvat_training_batched_50.xlsx", index=False)

training_df.to_excel("mvat_training_batched_50.xlsx", index=False)

from sentence_transformers import SentenceTransformer, util
import pandas as pd

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_mvat_training_data_batched(df, sample_size=50):
    sample = df.sample(n=sample_size, random_state=42).reset_index(drop=False)
    prompts = sample["prompt"].tolist()
    responses_a = sample["response_a"].tolist()
    responses_b = sample["response_b"].tolist()

    # === Compute all embeddings in one batch ===
    all_texts = prompts + responses_a + responses_b
    all_embeddings = embedder.encode(all_texts, convert_to_tensor=True)

    training_rows = []
    for i, row in sample.iterrows():
        idx = row["index"]
        prompt = row["prompt"]
        resp_a = row["response_a"]
        resp_b = row["response_b"]
        label = row["LABEL"]

        # Offsets
        e_prompt = all_embeddings[i]
        e_a = all_embeddings[i + sample_size]
        e_b = all_embeddings[i + 2 * sample_size]

        # Cosine similarity
        sim_pa = util.cos_sim(e_prompt, e_a).item()
        sim_pb = util.cos_sim(e_prompt, e_b).item()
        sim_ab = util.cos_sim(e_a, e_b).item()

        # Originality
        orig_a = sim_pa - sim_ab
        orig_b = sim_pb - sim_ab

        training_rows.append({
            "Index": idx,
            "f_q": 1,
            "f_a": 1,
            "f_b": 1,
            "keyword_a": 1,
            "keyword_b": 1,
            "refusal_a": 0,
            "refusal_b": 0,
            "caps_a": 0,
            "caps_b": 0,
            "LABEL": label,
            "orig_a": orig_a,
            "orig_b": orig_b
        })

    return pd.DataFrame(training_rows)


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the data
df = pd.read_excel("mvat_training_batched_50.xlsx")

# Step 2: Define features and target
features = [
    "f_q", "f_a", "f_b",
    "keyword_a", "keyword_b",
    "refusal_a", "refusal_b",
    "caps_a", "caps_b",
    "orig_a", "orig_b"
]
X = df[features]
y = df["LABEL"]

# Step 3: Train/Test Split (optional here since it's just 50 rows)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("✅ Accuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


df_50 = pd.read_excel("mvat_training_batched_50.xlsx")

# Step 1: Compute feature differences (A - B)
diff_df = pd.DataFrame()
diff_df["keyword_diff"] = df_50["keyword_a"] - df_50["keyword_b"]
diff_df["refusal_diff"] = df_50["refusal_a"] - df_50["refusal_b"]
diff_df["caps_diff"] = df_50["caps_a"] - df_50["caps_b"]
diff_df["orig_diff"] = df_50["orig_a"] - df_50["orig_b"]
diff_df["LABEL"] = df_50["LABEL"]

# Step 2: Train/test split
X = diff_df.drop(columns=["LABEL"])
y = diff_df["LABEL"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train logistic regression
model = LogisticRegression(multi_class='multinomial', max_iter=1000)
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("🎯 Accuracy:", round(acc * 100, 2), "%")
print("\n🧠 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Show learned weights
print("\n📈 Learned weights:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {round(coef, 4)}")


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Compute feature differences
diff_df = pd.DataFrame()
diff_df["factuality_diff"] = df_50["f_a"] - df_50["f_b"]
diff_df["keyword_diff"] = df_50["keyword_a"] - df_50["keyword_b"]
diff_df["refusal_diff"] = df_50["refusal_a"] - df_50["refusal_b"]
diff_df["caps_diff"] = df_50["caps_a"] - df_50["caps_b"]
diff_df["orig_diff"] = df_50["orig_a"] - df_50["orig_b"]
diff_df["LABEL"] = df_50["LABEL"]

# Step 2: Train/test split
X = diff_df.drop(columns=["LABEL"])
y = diff_df["LABEL"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("🎯 Accuracy:", round(acc * 100, 2), "%")
print("\n🧠 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Show learned weights
print("\n📈 Learned weights:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {round(coef, 4)}")


# === Extract 500-row MVAT training data with full features ===
df_500 = extract_mvat_training_data_batched_full(df, sample_size=500)

# === Compute A-B feature differences ===
diff_df_500 = pd.DataFrame()
diff_df_500["factuality_diff"] = df_500["f_a"] - df_500["f_b"]
diff_df_500["keyword_diff"] = df_500["keyword_a"] - df_500["keyword_b"]
diff_df_500["refusal_diff"] = df_500["refusal_a"] - df_500["refusal_b"]
diff_df_500["caps_diff"] = df_500["caps_a"] - df_500["caps_b"]
diff_df_500["orig_diff"] = df_500["orig_a"] - df_500["orig_b"]
diff_df_500["LABEL"] = df_500["LABEL"]

# === Train/test split ===
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X = diff_df_500.drop(columns=["LABEL"])
y = diff_df_500["LABEL"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train logistic regression ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === Evaluate predictions ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# === Output ===
print("🎯 Accuracy:", round(accuracy * 100, 2), "%")
print("\n📊 Classification Report:\n", report)

print("\n📈 Learned weights:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {round(coef, 4)}")


def extract_mvat_training_data_batched_full(df, sample_size=500):
    sample = df.sample(n=sample_size, random_state=42).reset_index(drop=False)
    prompts = sample["prompt"].tolist()
    responses_a = sample["response_a"].tolist()
    responses_b = sample["response_b"].tolist()
    indices = sample["index"].tolist()
    labels = sample["LABEL"].tolist()

    all_texts = prompts + responses_a + responses_b
    embeddings = embedder.encode(all_texts, convert_to_tensor=True)

    training_rows = []
    for i in range(sample_size):
        idx = indices[i]
        prompt = prompts[i]
        resp_a = responses_a[i]
        resp_b = responses_b[i]
        label = labels[i]

        emb_p = embeddings[i]
        emb_a = embeddings[i + sample_size]
        emb_b = embeddings[i + 2 * sample_size]

        sim_pa = util.cos_sim(emb_p, emb_a).item()
        sim_pb = util.cos_sim(emb_p, emb_b).item()
        sim_ab = util.cos_sim(emb_a, emb_b).item()

        row = {
            "Index": idx,
            "f_q": 1,
            "f_a": 1,
            "f_b": 1,
            "keyword_a": evaluate_keyword_salience(prompt, resp_a),
            "keyword_b": evaluate_keyword_salience(prompt, resp_b),
            "refusal_a": red_flag_refusal(resp_a),
            "refusal_b": red_flag_refusal(resp_b),
            "caps_a": red_flag_all_caps(resp_a),
            "caps_b": red_flag_all_caps(resp_b),
            "orig_a": sim_pa - sim_ab,
            "orig_b": sim_pb - sim_ab,
            "LABEL": label
        }

        training_rows.append(row)

    return pd.DataFrame(training_rows)


# === Extract 500-row MVAT training data with full features ===
df_500 = extract_mvat_training_data_batched_full(df, sample_size=500)

# === Compute A-B feature differences ===
diff_df_500 = pd.DataFrame()
diff_df_500["factuality_diff"] = df_500["f_a"] - df_500["f_b"]
diff_df_500["keyword_diff"] = df_500["keyword_a"] - df_500["keyword_b"]
diff_df_500["refusal_diff"] = df_500["refusal_a"] - df_500["refusal_b"]
diff_df_500["caps_diff"] = df_500["caps_a"] - df_500["caps_b"]
diff_df_500["orig_diff"] = df_500["orig_a"] - df_500["orig_b"]
diff_df_500["LABEL"] = df_500["LABEL"]

# === Train/test split ===
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X = diff_df_500.drop(columns=["LABEL"])
y = diff_df_500["LABEL"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train logistic regression ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === Evaluate predictions ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# === Output ===
print("🎯 Accuracy:", round(accuracy * 100, 2), "%")
print("\n📊 Classification Report:\n", report)

print("\n📈 Learned weights:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {round(coef, 4)}")


def extract_mvat_training_data_batched_full(df, sample_size=500):
    sample = df.sample(n=sample_size, random_state=42, replace=True).reset_index(drop=False)
    prompts = sample["prompt"].tolist()
    responses_a = sample["response_a"].tolist()
    responses_b = sample["response_b"].tolist()
    indices = sample["index"].tolist()
    labels = sample["LABEL"].tolist()

    all_texts = prompts + responses_a + responses_b
    embeddings = embedder.encode(all_texts, convert_to_tensor=True)

    training_rows = []
    for i in range(sample_size):
        idx = indices[i]
        prompt = prompts[i]
        resp_a = responses_a[i]
        resp_b = responses_b[i]
        label = labels[i]

        emb_p = embeddings[i]
        emb_a = embeddings[i + sample_size]
        emb_b = embeddings[i + 2 * sample_size]

        sim_pa = util.cos_sim(emb_p, emb_a).item()
        sim_pb = util.cos_sim(emb_p, emb_b).item()
        sim_ab = util.cos_sim(emb_a, emb_b).item()

        row = {
            "Index": idx,
            "f_q": 1,
            "f_a": 1,
            "f_b": 1,
            "keyword_a": evaluate_keyword_salience(prompt, resp_a),
            "keyword_b": evaluate_keyword_salience(prompt, resp_b),
            "refusal_a": red_flag_refusal(resp_a),
            "refusal_b": red_flag_refusal(resp_b),
            "caps_a": red_flag_all_caps(resp_a),
            "caps_b": red_flag_all_caps(resp_b),
            "orig_a": sim_pa - sim_ab,
            "orig_b": sim_pb - sim_ab,
            "LABEL": label
        }

        training_rows.append(row)

    return pd.DataFrame(training_rows)


# === Extract 500-row MVAT training data with full features ===
df_500 = extract_mvat_training_data_batched_full(df, sample_size=500)

# === Compute A-B feature differences ===
diff_df_500 = pd.DataFrame()
diff_df_500["factuality_diff"] = df_500["f_a"] - df_500["f_b"]
diff_df_500["keyword_diff"] = df_500["keyword_a"] - df_500["keyword_b"]
diff_df_500["refusal_diff"] = df_500["refusal_a"] - df_500["refusal_b"]
diff_df_500["caps_diff"] = df_500["caps_a"] - df_500["caps_b"]
diff_df_500["orig_diff"] = df_500["orig_a"] - df_500["orig_b"]
diff_df_500["LABEL"] = df_500["LABEL"]

# === Train/test split ===
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X = diff_df_500.drop(columns=["LABEL"])
y = diff_df_500["LABEL"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train logistic regression ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === Evaluate predictions ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# === Output ===
print("🎯 Accuracy:", round(accuracy * 100, 2), "%")
print("\n📊 Classification Report:\n", report)

print("\n📈 Learned weights:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {round(coef, 4)}")

print(df.columns.tolist())


import pandas as pd

df = pd.read_csv("train_features.csv")
df.columns = df.columns.str.strip()  # Clean whitespace from column names
print("✅ Columns:", df.columns.tolist())


!pip install sentence-transformers



import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import torch

# === Load the dataset ===
df = pd.read_csv("train_features.csv")
df.columns = df.columns.str.strip()

# === Load embedding model ===
embed_model = torch.hub.load('sentence-transformers/all-MiniLM-L6-v2', 'model')  # This assumes you already have it
tokenizer = torch.hub.load('sentence-transformers/all-MiniLM-L6-v2', 'tokenizer')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

# === Heuristics ===
def red_flag_all_caps(text):
    return int(text.isupper())

def red_flag_refusal(text):
    text = text.lower()
    return int("i can't" in text or "as an ai" in text or "i am not able" in text)

def evaluate_keyword_salience(prompt, response):
    prompt_words = set([word.lower() for word in prompt.split() if word.isalpha()])
    response_words = set([word.lower() for word in response.split() if word.isalpha()])
    return int(len(prompt_words & response_words) > 0)

# === Extract 500 rows ===
sample = df.sample(n=500, random_state=42, replace=True).reset_index(drop=True)

rows = []
for i in range(len(sample)):
    row = sample.iloc[i]
    prompt = row["prompt"]
    a = row["response_a"]
    b = row["response_b"]
    label = row["LABEL"]

    # Embeddings
    try:
        emb_p = get_embedding(prompt)
        emb_a = get_embedding(a)
        emb_b = get_embedding(b)
        sim_pa = cosine_similarity(emb_p, emb_a)[0][0]
        sim_pb = cosine_similarity(emb_p, emb_b)[0][0]
        sim_ab = cosine_similarity(emb_a, emb_b)[0][0]
        orig_a = sim_pa - sim_ab
        orig_b = sim_pb - sim_ab
    except Exception as e:
        orig_a = np.nan
        orig_b = np.nan

    rows.append({
        "Index": row["Column1"],
        "f_q": 1,
        "f_a": 1,
        "f_b": 1,
        "keyword_a": evaluate_keyword_salience(prompt, a),
        "keyword_b": evaluate_keyword_salience(prompt, b),
        "refusal_a": red_flag_refusal(a),
        "refusal_b": red_flag_refusal(b),
        "caps_a": red_flag_all_caps(a),
        "caps_b": red_flag_all_caps(b),
        "orig_a": orig_a,
        "orig_b": orig_b,
        "LABEL": label
    })

# === Save to Excel ===
final_df = pd.DataFrame(rows)
final_df.to_excel("mvat_training_500.xlsx", index=False)
print("✅ 500-row MVAT file saved as mvat_training_500.xlsx")


!pip install --force-reinstall --no-cache-dir --upgrade numpy scipy scikit-learn torch


import pandas as pd
df = pd.read_csv("train_features.csv")


import pandas as pd
import numpy as np
import re
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util
import spacy

# === Load embedding + NLP models ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

# === Red flag detectors ===
def red_flag_all_caps(text):
    words = re.findall(r'\b[A-Z]{2,}\b', text)
    return int(len(words) > 5)

def red_flag_refusal(text):
    return int(any(phrase in text.lower() for phrase in [
        "as an ai", "i am unable", "i cannot", "i'm sorry", "i do not have", "i don't have"
    ]))

# === Keyword salience ===
def keyword_salience(prompt, response):
    prompt_doc = nlp(prompt)
    keywords = [token.lemma_.lower() for token in prompt_doc if token.pos_ in ["NOUN", "PROPN"]]
    if not keywords:
        return 0
    first_half = response[:len(response)//2].lower()
    return int(any(kw in first_half for kw in keywords))

# === Factuality mock flags ===
def mock_factuality_flags(prompt, response_a, response_b):
    # Treat all as factual unless refusal is detected
    f_q = int("?" in prompt or "how" in prompt.lower() or "what" in prompt.lower())
    f_a = 0 if red_flag_refusal(response_a) else 1
    f_b = 0 if red_flag_refusal(response_b) else 1
    return f_q, f_a, f_b

# === Embedding-based originality ===
def originality_scores(prompt, response_a, response_b):
    embs = embedder.encode([prompt, response_a, response_b], convert_to_tensor=True)
    sim_pa = util.cos_sim(embs[0], embs[1]).item()
    sim_pb = util.cos_sim(embs[0], embs[2]).item()
    sim_ab = util.cos_sim(embs[1], embs[2]).item()
    orig_a = sim_pa - sim_ab
    orig_b = sim_pb - sim_ab
    return orig_a, orig_b

# === Batch generator ===
def extract_mvat_training_data(df, sample_size=500):
    rows = []
    sample = df.sample(n=sample_size, random_state=42, replace=True).reset_index(drop=True)

    for i, row in sample.iterrows():
        prompt = row["prompt"]
        ra = row["response_a"]
        rb = row["response_b"]
        label = row["LABEL"]

        f_q, f_a, f_b = mock_factuality_flags(prompt, ra, rb)
        k_a = keyword_salience(prompt, ra)
        k_b = keyword_salience(prompt, rb)
        rfa = red_flag_refusal(ra)
        rfb = red_flag_refusal(rb)
        ca = red_flag_all_caps(ra)
        cb = red_flag_all_caps(rb)
        orig_a, orig_b = originality_scores(prompt, ra, rb)

        rows.append({
            "Index": row["Column1"],
            "f_q": f_q,
            "f_a": f_a,
            "f_b": f_b,
            "keyword_a": k_a,
            "keyword_b": k_b,
            "refusal_a": rfa,
            "refusal_b": rfb,
            "caps_a": ca,
            "caps_b": cb,
            "LABEL": label,
            "orig_a": orig_a,
            "orig_b": orig_b
        })

    return pd.DataFrame(rows)

# === Run and save ===
df_500 = extract_mvat_training_data(df, sample_size=500)
df_500.to_excel("mvat_training_500.xlsx", index=False)


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load 500-row file
df_500 = pd.read_excel("mvat_training_500.xlsx")

# Compute feature differences (A - B)
diff_df = pd.DataFrame()
diff_df["factuality_diff"] = df_500["f_a"] - df_500["f_b"]
diff_df["keyword_diff"] = df_500["keyword_a"] - df_500["keyword_b"]
diff_df["refusal_diff"] = df_500["refusal_a"] - df_500["refusal_b"]
diff_df["caps_diff"] = df_500["caps_a"] - df_500["caps_b"]
diff_df["orig_diff"] = df_500["orig_a"] - df_500["orig_b"]

# Use only rows that aren't ties (label ≠ 2)
filtered = df_500[df_500["LABEL"] != 2].copy()
X = diff_df.loc[filtered.index]
y = filtered["LABEL"]

# Train model
clf = LogisticRegression()
clf.fit(X, y)

# Evaluate
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(f"🎯 Accuracy: {acc * 100:.1f} %\n")
print("🧠 Classification Report:")
print(classification_report(y, y_pred, digits=3))

# Print weights
weights = dict(zip(X.columns, clf.coef_[0]))
print("\n📈 Learned weights:")
for k, v in weights.items():
    print(f"{k}: {round(v, 4)}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

# === Load the 500-row feature file ===
df_500 = pd.read_excel("mvat_training_500.xlsx")

# === Compute A-B feature differences ===
diff_df_500 = pd.DataFrame()
diff_df_500["factuality_diff"] = df_500["f_a"] - df_500["f_b"]
diff_df_500["keyword_diff"]    = df_500["keyword_a"] - df_500["keyword_b"]
diff_df_500["refusal_diff"]    = df_500["refusal_a"] - df_500["refusal_b"]
diff_df_500["caps_diff"]       = df_500["caps_a"] - df_500["caps_b"]
diff_df_500["orig_diff"]       = df_500["orig_a"] - df_500["orig_b"]

# === Target label ===
y = df_500["LABEL"]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(diff_df_500, y, test_size=0.3, random_state=42)

# === Random Forest Classifier ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === Predict + Report ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"🎯 Accuracy: {round(acc * 100, 2)} %\n")
print("🧠 Classification Report:")
print(classification_report(y_test, y_pred))


import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === Define features and target ===
features = ["factuality_diff", "keyword_diff", "refusal_diff", "caps_diff", "orig_diff"]
X = diff_df_500[features]
y = df_500["LABEL"]

# === Split into train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Train XGBoost Classifier ===
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# === Predict and evaluate ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("🎯 Accuracy:", round(acc * 100, 2), "%")
print("\n🧠 Classification Report:\n", classification_report(y_test, y_pred))

# === Show learned feature importances ===
importances = dict(zip(features, model.feature_importances_))
print("\n📈 Learned feature importances:")
for k, v in importances.items():
    print(f"{k}: {round(v, 4)}")


import pandas as pd

# Load the full dataset
df = pd.read_csv("train_features.csv")
df.columns = df.columns.str.strip()  # Strip whitespace from headers

# Shuffle the data randomly
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Slice into 3 groups of 500 rows
mvat_df = df_shuffled.iloc[:500].copy()
distilbert_df = df_shuffled.iloc[500:1000].copy()
test_df = df_shuffled.iloc[1000:1500].copy()

# Export each to CSV
mvat_df.to_csv("mvat_train_500.csv", index=False)
distilbert_df.to_csv("distilbert_train_500.csv", index=False)
test_df.to_csv("test_500.csv", index=False)

print("✅ Three 500-row files saved:")
print("- mvat_train_500.csv")
print("- distilbert_train_500.csv")
print("- test_500.csv")


import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

# === Load data ===
df = pd.read_excel("mvat_training_distilbert_500.xlsx")
df.columns = df.columns.str.strip()  # Clean any whitespace in headers

# === Combine prompt, response_a, response_b into strings ===
df['text'] = df.apply(
    lambda row: f"Prompt: {row['prompt']}\nResponse A: {row['response_a']}\nResponse B: {row['response_b']}", axis=1
)

# === Tokenize with HuggingFace DistilBERT tokenizer ===
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize all text samples with truncation and padding
tokenized = tokenizer(
    df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors='pt'
)

# Extract input_ids and attention_mask
input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']
labels = df['LABEL'].tolist()

# === Output shapes for confirmation ===
print("✅ Tokenization complete")
print("🧾 Input shape:", input_ids.shape)
print("🔖 Labels length:", len(labels))


# === SETUP ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Force CPU usage
device = torch.device("cpu")

# === LOAD DATA ===
df = pd.read_csv("distilbert_500.csv")  # must have prompt, response_a, response_b, LABEL

# === PREPARE TEXT INPUTS ===
df["text_pair"] = df["prompt"] + " [SEP] " + df["response_a"] + " [SEP] " + df["response_b"]

# === TRAIN/TEST SPLIT ===
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text_pair"].tolist(), df["LABEL"].tolist(), test_size=0.3, random_state=42
)

# === TOKENIZE ===
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# === WRAP IN TORCH DATASET ===
class MVATDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = MVATDataset(train_encodings, train_labels)
val_dataset = MVATDataset(val_encodings, val_labels)

# === LOAD DISTILBERT WITH CLASSIFIER HEAD ===
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3).to(device)

# === TRAINING CONFIG ===
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_steps=10,
    save_strategy="no",
    load_best_model_at_end=False,
    seed=42
)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# === RUN TRAINING ===
trainer.train()

# === EVALUATE ===
preds = trainer.predict(val_dataset)
y_pred = preds.predictions.argmax(axis=1)
y_true = val_labels

print(f"\n🎯 Accuracy: {accuracy_score(y_true, y_pred) * 100:.1f} %\n")
print("🧠 Classification Report:")
print(classification_report(y_true, y_pred))


print("✅ Notebook is alive")

import pandas as pd

df = pd.read_csv("distilbert_500.csv")
print(df.columns.tolist())
df.head()


pip install transformers


import pandas as pd

# Load the CSV file
df = pd.read_csv('distilbert_500.csv')

# Check the first few rows to ensure it loaded correctly
df.head()



from transformers import DistilBertTokenizer

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the data
def tokenize_data(row):
    return tokenizer(
        row['prompt'], 
        row['response_a'], 
        row['response_b'], 
        padding=True, 
        truncation=True, 
        max_length=512
    )

# Apply the tokenizer to the dataset
tokenized_data = df.apply(tokenize_data, axis=1)

# Check a sample tokenized output
tokenized_data.head()


from transformers import DistilBertTokenizer

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the data
def tokenize_data(row):
    return tokenizer(
        row['prompt'], 
        row['response_a'], 
        row['response_b'], 
        padding=True, 
        truncation=True, 
        max_length=512
    )

# Apply the tokenizer to the dataset
tokenized_data = df.apply(tokenize_data, axis=1)

# Check a sample tokenized output
tokenized_data.head()


import pandas as pd

# Load the CSV file
df = pd.read_csv('distilbert_500.csv')

# Check the first few rows to ensure it loaded correctly
df.head()

from transformers import DistilBertTokenizer

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the data
def tokenize_data(row):
    return tokenizer(
        row['prompt'], 
        row['response_a'], 
        row['response_b'], 
        padding=True, 
        truncation=True, 
        max_length=512
    )

# Apply the tokenizer to the dataset
tokenized_data = df.apply(tokenize_data, axis=1)

# Check a sample tokenized output
tokenized_data.head()


# Tokenize the data with explicit handling of overflow and padding
def tokenize_data(row):
    return tokenizer(
        row['prompt'], 
        row['response_a'], 
        row['response_b'], 
        padding='max_length',  # Ensure padding to max length
        truncation=True,       # Truncate if the sequence exceeds max length
        max_length=512        # Limit to 512 tokens
    )

# Apply the tokenizer to the dataset
tokenized_data = df.apply(tokenize_data, axis=1)

# Check a sample tokenized output
tokenized_data.head()


# Tokenize the data with explicit handling of overflow and padding
def tokenize_data(row):
    return tokenizer(
        row['prompt'], 
        row['response_a'], 
        row['response_b'], 
        padding='max_length',  # Ensure padding to max length
        truncation=True,       # Truncate if the sequence exceeds max length
        max_length=512,        # Limit to 512 tokens
        return_tensors="pt"    # Return tensors directly
    )

# Apply the tokenizer to the dataset
tokenized_data = df.apply(tokenize_data, axis=1)

# Check a sample tokenized output
tokenized_data.head()


# Tokenize the data with explicit handling of overflow and padding
def tokenize_data(row):
    return tokenizer(
        row['prompt'], 
        row['response_a'], 
        row['response_b'], 
        padding='max_length',  # Ensure padding to max length
        truncation='only_second',  # Truncate only the second sequence if needed
        max_length=512,           # Limit to 512 tokens
        return_tensors="pt"       # Return tensors directly
    )

# Apply the tokenizer to the dataset
tokenized_data = df.apply(tokenize_data, axis=1)

# Check a sample tokenized output
tokenized_data.head()


# Function to truncate prompt and responses to 50 characters
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Apply truncation before tokenization
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])
    response_a = truncate_text(row['response_a'])
    response_b = truncate_text(row['response_b'])
    
    return tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length', 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )

# Apply the tokenizer to the dataset
tokenized_data = df.apply(tokenize_data, axis=1)

# Check a sample tokenized output
tokenized_data.head()


import torch

# Prepare the dataset with tensors
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Check the shapes of the resulting tensors
print(input_ids.shape, attention_mask.shape, labels.shape)


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load the DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Split the data into train and validation sets (80/20 split)
from sklearn.model_selection import train_test_split

train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training

    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)
# Start the training
trainer.train()


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load the DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Split the data into train and validation sets (80/20 split)
from sklearn.model_selection import train_test_split

train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Start the training
trainer.train()




from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load the DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Split the data into train and validation sets (80/20 split)
from sklearn.model_selection import train_test_split

train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Start the training
trainer.train()


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load the DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Split the data into train and validation sets (80/20 split)
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Start the training
trainer.train()


pip show accelerate


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load the DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Split the data into train and validation sets (80/20 split)
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Start the training
trainer.train()


import pandas as pd
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Reload the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the dataset
def tokenize_data(row):
    return tokenizer(
        row['prompt'], 
        row['response_a'], 
        row['response_b'], 
        padding='max_length', 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )

tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Now you can proceed with training the model


# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])
    response_a = truncate_text(row['response_a'])
    response_b = truncate_text(row['response_b'])
    
    return tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length', 
        truncation=True, 
        max_length=512,       # Limit to 512 tokens
        return_tensors="pt"
    )

tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)


# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])
    response_a = truncate_text(row['response_a'])
    response_b = truncate_text(row['response_b'])
    
    return tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length', 
        truncation=True, 
        max_length=512,       # Limit to 512 tokens
        return_tensors="pt"
    )

tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)


import pandas as pd
from transformers import DistilBertTokenizer
import torch
from sklearn.model_selection import train_test_split

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])
    response_a = truncate_text(row['response_a'])
    response_b = truncate_text(row['response_b'])
    
    return tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length', 
        truncation=True, 
        max_length=512,       # Limit to 512 tokens
        return_tensors="pt"
    )

# Apply the tokenizer to the dataset
tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)


import pandas as pd
from transformers import DistilBertTokenizer
import torch
from sklearn.model_selection import train_test_split

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])
    response_a = truncate_text(row['response_a'])
    response_b = truncate_text(row['response_b'])
    
    return tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length', 
        truncation=True, 
        max_length=512,       # Limit to 512 tokens
        return_tensors="pt"
    )

# Apply the tokenizer to the dataset
tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)


import pandas as pd
from transformers import DistilBertTokenizer
import torch
from sklearn.model_selection import train_test_split

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Check the first few rows of the dataset
print("Dataset loaded. Here's a preview:")
print(df.head())

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])
    response_a = truncate_text(row['response_a'])
    response_b = truncate_text(row['response_b'])
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length', 
        truncation=True, 
        max_length=512,       # Limit to 512 tokens
        return_tensors="pt"
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)

# Check tokenized data
print("Tokenized data sample:")
print(tokenized_data.head())

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Print shapes to verify
print(f"input_ids shape: {input_ids.shape}")
print(f"attention_mask shape: {attention_mask.shape}")
print(f"labels shape: {labels.shape}")

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Print confirmation of DataLoader creation
print("DataLoader ready for training:")
print(f"Train DataLoader: {len(train_dataloader)} batches")
print(f"Validation DataLoader: {len(val_dataloader)} batches")


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluate every epoch
    save_strategy="epoch",           # save model every epoch
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Start the training process
trainer.train()


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Set up the training arguments with evaluation strategy and saving model every epoch
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluate at the end of each epoch
    save_strategy="epoch",           # save model at the end of each epoch
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Start the training process
trainer.train()


pip install --upgrade transformers


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Set up the training arguments with evaluation strategy and saving model every epoch
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluate at the end of each epoch
    save_strategy="epoch",           # save model at the end of each epoch
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Start the training process
trainer.train()


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Set up the training arguments without evaluation_strategy
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    save_strategy="epoch",           # save model at the end of each epoch
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Start training and manually evaluate after each epoch
for epoch in range(training_args.num_train_epochs):
    print(f"Starting epoch {epoch + 1}")
    trainer.train()
    print(f"Evaluating after epoch {epoch + 1}")
    trainer.evaluate()


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Set up the training arguments without evaluation_strategy
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    save_strategy="epoch",           # save model at the end of each epoch
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Start training and manually evaluate after each epoch
for epoch in range(training_args.num_train_epochs):
    print(f"Starting epoch {epoch + 1}")
    trainer.train()
    print(f"Evaluating after epoch {epoch + 1}")
    trainer.evaluate()


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])
    response_a = truncate_text(row['response_a'])
    response_b = truncate_text(row['response_b'])
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length', 
        truncation=True, 
        max_length=512,       # Limit to 512 tokens
        return_tensors="pt"
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])
    response_a = truncate_text(row['response_a'])
    response_b = truncate_text(row['response_b'])
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length', 
        truncation=True, 
        max_length=512,       # Limit to 512 tokens
        return_tensors="pt"
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Print shapes to verify
print(f"input_ids shape: {input_ids.shape}")
print(f"attention_mask shape: {attention_mask.shape}")
print(f"labels shape: {labels.shape}")

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    save_strategy="epoch",           # save model at the end of each epoch
)

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Start training and manually evaluate after each epoch
for epoch in range(training_args.num_train_epochs):
    print(f"Starting epoch {epoch + 1}")
    trainer.train()
    print(f"Evaluating after epoch {epoch + 1}")
    trainer.evaluate()


import pandas as pd  # Import pandas again
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])
    response_a = truncate_text(row['response_a'])
    response_b = truncate_text(row['response_b'])
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length', 
        truncation=True, 
        max_length=512,       # Limit to 512 tokens
        return_tensors="pt"
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Print shapes to verify
print(f"input_ids shape: {input_ids.shape}")
print(f"attention_mask shape: {attention_mask.shape}")
print(f"labels shape: {labels.shape}")

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',_


import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])
    response_a = truncate_text(row['response_a'])
    response_b = truncate_text(row['response_b'])
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length', 
        truncation=True, 
        max_length=512,  # Limit to 512 tokens
        return_tensors="pt"
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Print shapes to verify
print(f"input_ids shape: {input_ids.shape}")
print(f"attention_mask shape: {attention_mask.shape}")
print(f"labels shape: {labels.shape}")

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels_


# Import necessary libraries
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Step 1: Load your dataset (make sure your CSV is in the working directory)
df = pd.read_csv('distilbert_500.csv')

# Step 2: Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Step 3: Define a truncation function (for token length limits)
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Step 4: Tokenize the data (prompt, response_a, response_b)
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])  # Truncate the prompt text
    response_a = truncate_text(row['response_a'])  # Truncate response_a text
    response_b = truncate_text(row['response_b'])  # Truncate response_b text
    
    return tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length',  # Ensure max-length padding
        truncation=True,  # Truncate texts that exceed max length
        max_length=512,  # Max token length for DistilBERT
        return_tensors="pt"  # Return tensors for PyTorch_


import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])  # Truncate the prompt text
    response_a = truncate_text(row['response_a'])  # Truncate response_a text
    response_b = truncate_text(row['response_b'])  # Truncate response_b text
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length',  # Ensure max-length padding
        truncation=True,  # Truncate texts that exceed max length
        max_length=512,  # Max token length for DistilBERT
        return_tensors="pt"  # Return tensors for PyTorch
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data


import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])  # Truncate the prompt text
    response_a = truncate_text(row['response_a'])  # Truncate response_a text
    response_b = truncate_text(row['response_b'])  # Truncate response_b text
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length',  # Ensure max-length padding
        truncation=True,  # Truncate texts that exceed max length
        max_length=512,  # Max token length for DistilBERT
        return_tensors="pt"  # Return tensors for PyTorch
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)  # <-- Added closing parenthesis here

# Extract the tokenized data into input_ids, attention_mask, and


import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])  # Truncate the prompt text
    response_a = truncate_text(row['response_a'])  # Truncate response_a text
    response_b = truncate_text(row['response_b'])  # Truncate response_b text
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length',  # Ensure max-length padding
        truncation=True,  # Truncate texts that exceed max length
        max_length=512,  # Max token length for DistilBERT
        return_tensors="pt"  # Return tensors for PyTorch
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)  # <-- Added closing parenthesis here

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Print shapes to verify
print(f"input_ids shape: {input_ids.shape}")
print(f"attention_mask shape: {attention_mask.shape}")
print(f"labels shape: {labels.shape}")

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,      # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,        # number of warmup steps for learning rate scheduler
    weight_decay=0.01,       # strength of weight decay
    logging_dir='./logs',    # directory for storing logs
    logging_steps=10,
    save_strategy="epoch",   # save model at the end of each epoch
)

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Start training and manually evaluate after each epoch
for epoch in range(training_args.num_train_epochs):
    print(f"Starting epoch {epoch + 1}")
    trainer.train()  # Train the model
    print(f"Evaluating after epoch {epoch + 1}")
    trainer.evaluate()  # Evaluate the model


import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])  # Truncate the prompt text
    response_a = truncate_text(row['response_a'])  # Truncate response_a text
    response_b = truncate_text(row['response_b'])  # Truncate response_b text
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length',  # Ensure max-length padding
        truncation=True,  # Truncate texts that exceed max length
        max_length=512,  # Max token length for DistilBERT
        return_tensors="pt"  # Return tensors for PyTorch
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Print shapes to verify
print(f"input_ids shape: {input_ids.shape}")
print(f"attention_mask shape: {attention_mask.shape}")
print(f"labels shape: {labels.shape}")

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Prepare DataLoader for PyTorch
train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,      # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,        # number of warmup steps for learning rate scheduler
    weight_decay=0.01,       # strength of weight decay
    logging_dir='./logs',    # directory for storing logs
    logging_steps=10,
    save_strategy="epoch",   # save model at the end of each epoch
)

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Create the Trainer
trainer = Trainer(
    model=model,                         # The model to be trained
    args=training_args,                  # The training arguments
    train_dataset=train_dataset,         # The training dataset
    eval_dataset=val_dataset             # The validation dataset
)

# Start training and manually evaluate after each epoch
for epoch in range(training_args.num_train_epochs):
    print(f"Starting epoch {epoch + 1}")
    trainer.train()  # Train the model
    print(f"Evaluating after epoch {epoch + 1}")
    trainer.evaluate()  # Evaluate the model


import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])  # Truncate the prompt text
    response_a = truncate_text(row['response_a'])  # Truncate response_a text
    response_b = truncate_text(row['response_b'])  # Truncate response_b text
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length',  # Ensure max-length padding
        truncation=True,  # Truncate texts that exceed max length
        max_length=512,  # Max token length for DistilBERT
        return_tensors="pt"  # Return tensors for PyTorch
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Print shapes to verify
print(f"input_ids shape: {input_ids.shape}")
print(f"attention_mask shape: {attention_mask.shape}")
print(f"labels shape: {labels.shape}")

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Create datasets in dictionary format for Trainer
train_dataset = {
    'input_ids': train_inputs,
    'attention_mask': attention_mask[:len(train_inputs)],
    'labels': train_labels
}
val_dataset = {
    'input_ids': val_inputs,
    'attention_mask': attention_mask[:len(val_inputs)],
    'labels': val_labels
}

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,      # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,        # number of warmup steps for learning rate scheduler
    weight_decay=0.01,       # strength of weight decay
    logging_dir='./logs',    # directory for storing logs
    logging_steps=10,
    save_strategy="epoch",   # save model at the end of each epoch
)

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # the training dataset
    eval_dataset=val_dataset             # the validation dataset
)

# Start training and manually evaluate after each epoch
for epoch in range(training_args.num_train_epochs):
    print(f"Starting epoch {epoch + 1}")
    trainer.train()  # Train the model
    print(f"Evaluating after epoch {epoch + 1}")
    trainer.evaluate()  # Evaluate the model


import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])  # Truncate the prompt text
    response_a = truncate_text(row['response_a'])  # Truncate response_a text
    response_b = truncate_text(row['response_b'])  # Truncate response_b text
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length',  # Ensure max-length padding
        truncation=True,  # Truncate texts that exceed max length
        max_length=512,  # Max token length for DistilBERT
        return_tensors="pt"  # Return tensors for PyTorch
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Print shapes to verify
print(f"input_ids shape: {input_ids.shape}")
print(f"attention_mask shape: {attention_mask.shape}")
print(f"labels shape: {labels.shape}")

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Create DataLoader-ready TensorDataset
train_dataset = TensorDataset(train_inputs, attention_mask[:len(train_inputs)], train_labels)
val_dataset = TensorDataset(val_inputs, attention_mask[:len(val_inputs)], val_labels)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,      # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,        # number of warmup steps for learning rate scheduler
    weight_decay=0.01,       # strength of weight decay
    logging_dir='./logs',    # directory for storing logs
    logging_steps=10,
    save_strategy="epoch",   # save model at the end of each epoch
)

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # validation dataset
)

# Start training and manually evaluate after each epoch
for epoch in range(training_args.num_train_epochs):
    print(f"Starting epoch {epoch + 1}")
    trainer.train()  # Train the model
    print(f"Evaluating after epoch {epoch + 1}")
    trainer.evaluate()  # Evaluate the model


import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])  # Truncate the prompt text
    response_a = truncate_text(row['response_a'])  # Truncate response_a text
    response_b = truncate_text(row['response_b'])  # Truncate response_b text
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length',  # Ensure max-length padding
        truncation=True,  # Truncate texts that exceed max length
        max_length=512,  # Max token length for DistilBERT
        return_tensors="pt"  # Return tensors for PyTorch
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Print shapes to verify
print(f"input_ids shape: {input_ids.shape}")
print(f"attention_mask shape: {attention_mask.shape}")
print(f"labels shape: {labels.shape}")

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Create datasets in dictionary format for Trainer
train_dataset = {
    'input_ids': train_inputs,
    'attention_mask': attention_mask[:len(train_inputs)],
    'labels': train_labels
}
val_dataset = {
    'input_ids': val_inputs,
    'attention_mask': attention_mask[:len(val_inputs)],
    'labels': val_labels
}

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,      # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,        # number of warmup steps for learning rate scheduler
    weight_decay=0.01,       # strength of weight decay
    logging_dir='./logs',    # directory for storing logs
    logging_steps=10,
    save_strategy="epoch",   # save model at the end of each epoch
)

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # validation dataset
)

# Start training and manually evaluate after each epoch
for epoch in range(training_args.num_train_epochs):
    print(f"Starting epoch {epoch + 1}")
    trainer.train()  # Train the model
    print(f"Evaluating after epoch {epoch + 1}")
    trainer.evaluate()  # Evaluate the model


import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Reload the CSV file
df = pd.read_csv('distilbert_500.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to truncate text
def truncate_text(text, max_length=50):
    return text[:max_length] if isinstance(text, str) else text

# Tokenize the dataset with truncation before passing it to the tokenizer
def tokenize_data(row):
    prompt = truncate_text(row['prompt'])  # Truncate the prompt text
    response_a = truncate_text(row['response_a'])  # Truncate response_a text
    response_b = truncate_text(row['response_b'])  # Truncate response_b text
    
    tokenized = tokenizer(
        prompt, 
        response_a, 
        response_b, 
        padding='max_length',  # Ensure max-length padding
        truncation=True,  # Truncate texts that exceed max length
        max_length=512,  # Max token length for DistilBERT
        return_tensors="pt"  # Return tensors for PyTorch
    )
    return tokenized

# Apply the tokenizer to the dataset and collect the tokenized data
tokenized_data = df.apply(tokenize_data, axis=1)

# Extract the tokenized data into input_ids, attention_mask, and labels
input_ids = torch.cat([x['input_ids'] for x in tokenized_data], dim=0)
attention_mask = torch.cat([x['attention_mask'] for x in tokenized_data], dim=0)
labels = torch.tensor(df['LABEL'].values)

# Print shapes to verify
print(f"input_ids shape: {input_ids.shape}")
print(f"attention_mask shape: {attention_mask.shape}")
print(f"labels shape: {labels.shape}")

# Now split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Create datasets in dictionary format for Trainer (each batch is a dictionary)
train_dataset = [
    {"input_ids": train_inputs[i], "attention_mask": attention_mask[i], "labels": train_labels[i]} 
    for i in range(len(train_inputs))
]

val_dataset = [
    {"input_ids": val_inputs[i], "attention_mask": attention_mask[i], "labels": val_labels[i]} 
    for i in range(len(val_inputs))
]

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,      # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,        # number of warmup steps for learning rate scheduler
    weight_decay=0.01,       # strength of weight decay
    logging_dir='./logs',    # directory for storing logs
    logging_steps=10,
    save_strategy="epoch",   # save model at the end of each epoch
)

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # the training dataset
    eval_dataset=val_dataset             # the validation dataset
)

# Start training and manually evaluate after each epoch
for epoch in range(training_args.num_train_epochs):
    print(f"Starting epoch {epoch + 1}")
    trainer.train()  # Train the model
    print(f"Evaluating after epoch {epoch + 1}")
    trainer.evaluate()  # Evaluate the model



import json
import torch
import transformers
from tqdm import tqdm
from datasets import load_dataset


def load_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def read_passages(passages_file):
    with open(passages_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def retrieve_context(sample_id, passages_dict):
    if sample_id in passages_dict:
        return "\n".join(passages_dict[sample_id])
    return ""


def build_prompt(question_text, option_a, option_b, option_c, option_d, context):

    prompt = (
        f"You are a professional Turkish AI assistant. Use only the information from the context below to answer the question.\n"
        + f"Question:\n{question_text}\n\n"
        + f"Context:\n{context}\n\n"
        + f"Options:\n"
        + f"A) {option_a}\n"
        + f"B) {option_b}\n"
        + f"C) {option_c}\n"
        + f"D) {option_d}\n\n"
        + "Please answer with exactly one letter (A, B, C, or D).\nAnswer:"
    )
    return prompt


def generate_raw_text(prompt, model, tokenizer, device, max_new_tokens=10):

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


import re

def extract_letter(answer_text):
    # find the last occurrence of "Answer:"
    # and extract what comes *after* that
    pattern = r'\bAnswer:\s*([A-D])(?:\)|\.|$|\s)'
    match = re.search(pattern, answer_text)
    if match:
        return match.group(1).upper()  # A, B, C, or D

    # If no match is found
    first_letter_match = re.search(r'\b[A-D]\b', answer_text)
    if first_letter_match:
        return first_letter_match.group(0).upper()  # A, B, C, or D



def check_correctness(pred_choice, gold_letter):

    return (pred_choice == gold_letter.upper())


def main():

    model, tokenizer, device = load_model("meta-llama/Meta-Llama-3-8B-Instruct")
    dataset = load_dataset("CohereForAI/Global-MMLU", "tr", split="test")
    passages_dict = read_passages("Turkisch.json")

    total_questions = 0
    correct_count = 0
    results = []

    for sample in tqdm(dataset, desc="Processing MMLU"):
        sample_id = sample["sample_id"]  # e.g. "abstract_algebra/test/0"
        question_text = sample["question"]
        option_a = sample["option_a"]
        option_b = sample["option_b"]
        option_c = sample["option_c"]
        option_d = sample["option_d"]
        gold_answer = sample["answer"]  # "A", "B", "C", or "D"

        if not question_text or not option_a or not option_b or not option_c or not option_d:
            continue
        context = retrieve_context(sample_id, passages_dict)

        # Build the prompt
        prompt = build_prompt(question_text, option_a, option_b, option_c, option_d, context)

        # raw text for check
        raw_model_output = generate_raw_text(prompt, model, tokenizer, device, max_new_tokens=10)

        last_answer_idx = raw_model_output.rfind("Answer:")
        if last_answer_idx != -1:
            substring_after_answer = raw_model_output[last_answer_idx+len("Answer:"):].strip()
        else:
            substring_after_answer = raw_model_output

        predicted_letter = extract_letter(substring_after_answer)

        # Evaluate
        is_correct = check_correctness(predicted_letter, gold_answer)
        if is_correct:
            correct_count += 1
        total_questions += 1

        # Save the result
        results.append({
            "sample_id": sample_id,
            "question": question_text,
            "options": {
                "A": option_a,
                "B": option_b,
                "C": option_c,
                "D": option_d
            },
            "gold_answer": gold_answer,
            "model_raw_output": raw_model_output,
            "parsed_substring": substring_after_answer,
            "predicted_letter": predicted_letter,
            "correct": is_correct,
            "context": context
        })

        print("------------------------------------------------------------")
        print(f"sample_id: {sample_id}")
        print(f"GOLD: {gold_answer}, PREDICTED: {predicted_letter}, CORRECT: {is_correct}")
        print(f"FULL MODEL OUTPUT:\n{raw_model_output}\n")

    # Summarize
    accuracy = correct_count / total_questions if total_questions > 0 else 0.0
    print(f"\nProcessed questions: {total_questions}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")

    # Save
    with open("global_mmlu_mc_results.json", "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)
    print("Results saved to global_mmlu_mc_results.json")


if __name__ == "__main__":
    main()

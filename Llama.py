import json
import torch
import transformers
from tqdm import tqdm
from datasets import load_dataset
import re


def load_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Load a pre-trained model and tokenizer, and set the device (GPU or CPU).

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the model, tokenizer, and device.
    """
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
    # gpu if available, otherwise cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def read_passages(passages_file):
    """Read passages from a JSON file.

    Args:
        passages_file (str): The path to the JSON file containing passages.

    Returns:
        dict: A dictionary with passages.
    """
    with open(passages_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def retrieve_context(sample_id, passages_dict):
    """Retrieve context passages for a given sample ID from a dictionary.

    Args:
        sample_id (str): The identifier for the sample.
        passages_dict (dict): A dictionary containing passages.

    Returns:
        str: The context passages joined by newline, or an empty string if not found.
    """
    if sample_id in passages_dict:
        return "\n".join(passages_dict[sample_id])
    return ""


def build_prompt(question_text, option_a, option_b, option_c, option_d, context):
    """Build a prompt for the model with the question, options, and context.

    Args:
        question_text (str): The question text.
        option_a (str): Option A.
        option_b (str): Option B.
        option_c (str): Option C.
        option_d (str): Option D.
        context (str): The context passages.

    Returns:
        str: The formatted prompt.
    """
    prompt = (
        f"You are a professional Greek AI assistant. Use only the information from the context below to answer the question.\n"
        + f"Question:\n{question_text}\n\n"
        + f"Context:\n{context}\n\n"
        + f"Options:\n"
        + f"A) {option_a}\n"
        + f"B) {option_b}\n"
        + f"C) {option_c}\n"
        + f"D) {option_d}\n\n"
        + f"Please answer with exactly one letter (A, B, C, or D).\nAnswer:"
    )
    return prompt


def generate_raw_text(prompt, model, tokenizer, device, max_new_tokens=10):
    """Generate raw text output from the model based on the prompt.

    Args:
        prompt (str): The input prompt.
        model: The pre-trained model.
        tokenizer: The tokenizer.
        device: The device to run the model on.
        max_new_tokens (int): The maximum number of new tokens to generate. Default 10.

    Returns:
        str: The generated text.
    """
    # tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,          # 10 tokens
            temperature=0.1,                        # deterministic
            do_sample=True,                         # sampling
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    # decode the generated tokens to text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_letter(answer_text):
    """Extract the predicted letter (A, B, C, or D) from the model's output.

    Args:
        answer_text (str): The text output from the model.

    Returns:
        str or None: The predicted letter.
    """
    # find the last occurrence of Answer:
    pattern = r'\bAnswer:\s*([A-D])(?:\)|\.|$|\s)'
    match = re.search(pattern, answer_text)
    if match:

        return match.group(1).upper()  # A, B, C, or D

    # If no match is found, find any standalone letter A, B, C, or D
    first_letter_match = re.search(r'\b[A-D]\b', answer_text)
    if first_letter_match:
        return first_letter_match.group(0).upper()  # A, B, C, or D


def check_correctness(pred_choice, gold_letter):
    """Check if the predicted choice matches the correct answer.

    Args:
        pred_choice (str): The predicted letter.
        gold_letter (str): The correct answer letter.

    Returns:
        bool: True if the prediction is correct, False otherwise.
    """
    return pred_choice == gold_letter.upper()


def main():
    """Process the dataset, generate answers using the model, and evaluate accuracy.

    This function loads the model, processes each sample in the dataset, generates answers,
    extracts the predicted letter, checks correctness, and saves the results.
    """
    model, tokenizer, device = load_model("meta-llama/Meta-Llama-3-8B-Instruct")
    dataset = load_dataset("CohereForAI/Global-MMLU", "el", split="test")
    passages_dict = read_passages("greek.json")

    total_questions = 0
    correct_count = 0
    results = []

    # loop through data with progress bar
    for sample in tqdm(dataset, desc="Processing MMLU"):
        sample_id = sample["sample_id"]  # e.g. "abstract_algebra/test/0"
        question_text = sample["question"]
        option_a = sample["option_a"]
        option_b = sample["option_b"]
        option_c = sample["option_c"]
        option_d = sample["option_d"]
        gold_answer = sample["answer"]  # "A", "B", "C", or "D"

        # skip samples with missing data
        if not question_text or not option_a or not option_b or not option_c or not option_d:
            continue
        context = retrieve_context(sample_id, passages_dict)

        # Build the prompt
        prompt = build_prompt(question_text, option_a, option_b, option_c, option_d, context)

        # get raw response
        raw_model_output = generate_raw_text(prompt, model, tokenizer, device, max_new_tokens=10)

        # get the text after the last "Answer:", hardly ever used in current version
        last_answer_idx = raw_model_output.rfind("Answer:")
        if last_answer_idx != -1:
            substring_after_answer = raw_model_output[last_answer_idx+len("Answer:"):].strip()
        else:
            substring_after_answer = raw_model_output

        predicted_letter = extract_letter(substring_after_answer)

        # check if prediction is right
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
        # Print the result for this sample to check
        print("------------------------------------------------------------")
        print(f"sample_id: {sample_id}")
        print(f"GOLD: {gold_answer}, PREDICTED: {predicted_letter}, CORRECT: {is_correct}")
        print(f"FULL MODEL OUTPUT:\n{raw_model_output}\n")

    # Calculate accuracy
    accuracy = correct_count / total_questions if total_questions > 0 else 0.0
    print(f"\nProcessed questions: {total_questions}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")

    # Save results to a JSON file
    with open("global_mmlu_mc_results.json", "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)
    print("Results saved to global_mmlu_mc_results.json")


if __name__ == "__main__":
    main()

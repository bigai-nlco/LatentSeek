"""
Data api
"""
from datasets import load_dataset, load_from_disk
from prompts import gsm8k_prompt, MATH_500_prompt, AIME_2024_prompt

def get_dataset(data_name_or_path, tokenizer, prompt_idx):
    """
    Args:
        data_name_or_path: dataset name or path
        tokenizer: tokenizer
        prompt_idx: which query prompt to use
    Returns:
        dataset: dataset
    """

    ### Load dataset ### 
    if "gsm8k" in data_name_or_path:
        try:
            dataset = load_from_disk(data_name_or_path)['test']
        except:
            dataset = load_dataset("openai/gsm8k", "socratic")["test"]
        question_col = "question"
        answer_col = "answer"

    elif "MATH-500" in data_name_or_path:
        try:
            dataset = load_from_disk(data_name_or_path)['test']
        except:
            dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        question_col = "problem"
        answer_col = "answer"

    elif "AIME_2024" in data_name_or_path:
        try:
            dataset = load_from_disk(data_name_or_path)
        except:
            dataset = load_dataset("Maxwell-Jia/AIME_2024")['train']
        question_col = "Problem"
        answer_col = "Answer"

    else:
        raise ValueError(f"Unsupported dataset: {data_name_or_path}")

    # preprocess dataset
    def preprocess_function(examples):
        '''
        Preprocess dataset

        Args:
            examples: dataset examples

        Returns:
            formatted: formatted dataset
        '''
        formatted = []
        questions = examples[question_col]
        for q in questions:
            if "gsm8k" in data_name_or_path:
                messages = gsm8k_prompt(q, prompt_idx)
            elif "MATH-500" in data_name_or_path:
                messages = MATH_500_prompt(q, prompt_idx)
            elif "AIME_2024" in data_name_or_path:
                messages = AIME_2024_prompt(q, prompt_idx)
            else:
                raise ValueError(f"Unsupported dataset: {data_name_or_path}")

            formatted.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        return {"formatted": formatted, "question": questions, "answer": examples[answer_col]}

    dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
    return dataset


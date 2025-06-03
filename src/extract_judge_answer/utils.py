import json
import re
from math_verify import parse, verify
from .grader import math_equal_process
from .math_equivalent_MATH import is_equiv
from .parse_utils_qwen import extract_answer as extract_fn
def extract_true_answer(text, name="gsm8k"):
    '''
    Extract answer from text

    Args:
        text: input text
        name: name of the dataset

    Returns:
        answer: extracted answer
    '''
    if "gsm8k" in name:
        label = text.split("#### ")[1]
        return label
    elif "MATH-500" in name:
        return text
    elif "AIME_2024" in name:
        return text
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def judge_answer(input, label, data_name="gsm8k", extract=True, prompt_idx=0):
    """Score.

    Judge whether the answer is correct or not.
    Only exact match is considered correct.

    Args:
        input (str): model response
        label (str): ground truth
        data_name (str): name of the dataset, ["gsm8k", "MATH-500"]
        extract (bool): whether to extract answer from model response
        prompt_idx (int): index of the solver prompt (different format) 

    Returns:
        bool: True if the answer is correct, False otherwise
    """
    if "gsm8k" in data_name:
        if extract:
            input = extract_answer(input, data_name="gsm8k", prompt_idx=prompt_idx)
        return (input == label)
    elif "MATH-500" in data_name:
        if extract:
            input = extract_answer(input, data_name="MATH-500", prompt_idx=prompt_idx)

        # huggingface math_verify
        hf_input = parse(input)
        hf_verifier_judge = verify(label, hf_input)
        if hf_verifier_judge:
            return True

        # qwen2.5-math 
        qwen_verifier_judge = math_equal_process((label, input))
        if qwen_verifier_judge:
            return True

        # exact match
        exact_judge = (str(input) == str(label))
        if exact_judge:
            return True

        # MATH-500
        MATH_500_judge = is_equiv(str(label), str(input))
        if MATH_500_judge:
            return True
        return False

    elif "AIME_2024" in data_name:
        if extract:
            input = extract_answer(input, data_name="AIME_2024", prompt_idx=prompt_idx)
            input = str(input)
            label = str(label)
        return (input == label)

    else:
        raise ValueError(f"Unknown dataset name: {data_name} for judge answer")
    
    
def extract_answer(text, data_name="gsm8k", prompt_idx=0, model_name="Qwen2.5-7B-Instruct"):
    '''
    Extract answer from model response

    Args:
        text: Raw response string from the language model
        data_name: name of the dataset, ["gsm8k", "MATH-500"]
        prompt_idx: index of the solver prompt (different format)

    Returns:
        answer: extracted answer(pure numbers)
    '''
    if "gsm8k" in data_name:
        if prompt_idx == 0:
            # 0: boxed
            if "qwen2.5-1.5b-instruct" in model_name.lower():
                # well, well, well
                temp = _extract_qwen25_1_5B_answer(text)
            else:
                temp = _extract_answer(text)
            return temp

        elif prompt_idx == 1:
            # 1: json
            try:
                answer = json.loads(text.strip('` \n'))
                final_answer = answer.get('final answer', '')
                if not isinstance(final_answer, str):
                    final_answer = str(final_answer)
                temp = _extract_answer(final_answer)
                return temp

            except json.JSONDecodeError:
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)[}<]'

                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL) 
                
                if match:
                    temp = _extract_answer(match.group(1))
                    return temp
                else:
                    temp = _extract_answer(text)
                    return temp


        else:
            raise ValueError(f"Unknown prompt index: {prompt_idx} for extract answer")

    elif "MATH-500" in data_name:
        if prompt_idx == 0:
            # 0: boxed
            temp = extract_fn(text, data_name='math')
            return temp

        elif prompt_idx == 1:
            # json
            try:
                answer = json.loads(text.strip('` \n'))
                final_answer = answer.get('final answer', '')
                if not isinstance(final_answer, str):
                    final_answer = str(final_answer)
                final_answer = final_answer.replace("\n", "")
                final_answer = final_answer.replace("\"", "")
                final_answer = final_answer.replace("\'", "")
                return final_answer

            except json.JSONDecodeError:
                text = text.replace("\n", "")
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)(}<|<\|)'


                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL) 
                
                if match:
                    temp = match.group(1)
                    temp = temp.replace("\n", "")
                    temp = temp.replace("\"", "")
                    temp = temp.replace("\'", "")
                    return temp
                else:
                    return None

    elif "AIME_2024" in data_name:
        if prompt_idx == 0:
            # 0: boxed
            temp = _extract_answer(text)
            return temp

        elif prompt_idx == 1:
            # 1: json, {"final answer": ...}
            try:
                answer = json.loads(text.strip('` \n'))
                final_answer = answer.get('final answer', '')
                if not isinstance(final_answer, str):
                    final_answer = str(final_answer)
                temp = _extract_answer(final_answer)
                return temp

            except json.JSONDecodeError:
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)[}<]'

                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL) 
                
                if match:
                    temp = _extract_answer(match.group(1))
                    return temp
                else:
                    temp = _extract_answer(text)
                    return temp


        else:
            raise ValueError(f"Unknown prompt index: {prompt_idx} for extract answer")
    else:
        raise ValueError(f"Unknown dataset name: {data_name} for extract answer")



######################
#       MATH         #
######################

def extract_MATH_solution(solution_str: str):
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        extracted final answer
    """""
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        processed_str = solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>.*?(\\boxed{.*}).*?</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        answer_pattern = r'\\boxed{(.*)}'
        matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        print("[Error] No valid answer tags found")
        return None
    final_answer = matches[-1].group(1).strip()
    return final_answer


def _extract_answer(text):
    """
    Extract numerical answer from generated text.
    handling various edge cases.
    
    Args:
        text (str): Generated text to extract answer from.
    
    Returns:
        str or None: Extracted numerical answer, or None if not found.
    """
    if text is None:
        return None
    
    text = text.strip()

    def clean_number(num_str):
        """Remove currency symbols, commas, and whitespace."""
        num_str = re.sub(r'[$€£¥]', '', num_str)
        num_str = re.sub(r',', '', num_str)
        num_str = re.sub(r'\s', '', num_str)
        return num_str

    ### Several Corner Cases ###
    # 1. \boxed{}
    boxed_pattern = r"\\boxed\{\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*\}"
    match = re.search(boxed_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))
    
    # 2. Answer:
    answer_pattern = r"Answer:\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    match = re.search(answer_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))
    
    # 3. =
    equals_pattern = r"=\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    match = re.search(equals_pattern, text)
    if match:
        return clean_number(match.group(1))

    # 4. With currency unit
    currency_pattern = r"is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*(?:dollars|euros|pounds|yen)"
    match = re.search(currency_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))

    # 5. Search from the last line of the text upwards, matching independent numbers
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line:
            final_num_pattern = r"([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*$"
            match = re.search(final_num_pattern, line)
            if match:
                return clean_number(match.group(1))

    # 6. Returns the last matching number in the text
    number_pattern = r"([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    matches = re.findall(number_pattern, text)
    if matches:
        return clean_number(matches[-1])

    return None


def _extract_qwen25_1_5B_answer(text):
    """
    Extract numerical answer from generated text for Qwen-2.5 1.5B model.
    handling various edge cases.

    Args:
        text (str): Generated text to extract answer from.

    Returns:
        str or None: Extracted numerical answer, or None if not found.
    """
    if text is None:
        return None

    text = text.strip()

    def clean_number(num_str):
        """Remove currency symbols, commas, and whitespace."""
        num_str = re.sub(r'[$€£¥]', '', num_str)
        num_str = re.sub(r',', '', num_str)
        num_str = re.sub(r'\s', '', num_str)
        return num_str

    ### Several Corner Cases ###
    # 1. \boxed{}
    boxed_pattern = r"\\boxed\{\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*\}"
    match = re.search(boxed_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))

    # 2. he answer is
    answer_pattern = r"he answer is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    match = re.search(answer_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))

    # 3. final answer is
    answer_pattern = r"final answer is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    match = re.search(answer_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))

    # 4. Returns the last matching number in the text
    number_pattern = r'\d+(?:,\d+)*(?:\.\d+)?'
    matches = re.findall(number_pattern, text)
    if matches:
        return clean_number(matches[-1])

    return None

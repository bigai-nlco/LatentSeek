"""
Solver prompts for different math datasets.
"""

def gsm8k_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
        prompt_idx (int): The index of the prompt to be used.

    """     
    prompt = [
            # idx 0: boxed
            [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": q},
                ],

            # idx 1: json
            [
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                {"role": "user", "content": 
                 f"QUESTION: {q} \n"
                 "Let's think step by step. "
                 "Please provide your thought process and your final answer separately and response in json format "
                 "containing the keys \"thought process\" and \"final answer\". "
                 "For example your response should be "
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 "Note that the final answer should be pure numbers, not the calculation formulas, and without any units or explanation!!! "}
                ],

    ]
    return prompt[prompt_idx] 


def MATH_500_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
    """
    prompt = [
            # idx 0: boxed
            [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": q},
                ],
            # idx 1: json
            [
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                {"role": "user", "content": 
                 f"QUESTION: {q} \n"
                 "Let's think step by step. "
                 "Please provide your thought process and your final answer separately and response in json format "
                 "containing the keys \"thought process\" and \"final answer\". "
                 "For example your response should be "
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 }
                ],
            ]
    return prompt[prompt_idx]


def AIME_2024_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
        prompt_idx (int): The index of the prompt to be used.

    """     
    prompt = [
            # idx 0: boxed
            [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": q},
                ],
            # idx 1: json 
            [
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                {"role": "user", "content": 
                 f"QUESTION: {q} \n"
                 "Let's think step by step. "
                 "Please provide your thought process and your final answer separately and response in json format "
                 "containing the keys \"thought process\" and \"final answer\". "
                 "For example your response should be "
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 "Note that the final answer should be pure numbers, not the calculation formulas, and without any units or explanation!!! "}
                ],


    ]
    return prompt[prompt_idx] 



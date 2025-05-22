import re
from termcolor import colored
from prompts.vera_prompts import get_vera_prompt
from prompts.vera_prompts import VERA_ANSWER_SYMBOL
import torch

class RewardModel(object):
    def __init__(
            self, 
            model,
            tokenizer,
            data_name: str = "gsm8k", 
            device: str = "cuda",
            rule_format_string: str = None,
        ):
        """
        Args:
            model: the model to use for reward prediction
            tokenizer: the tokenizer to use for reward prediction
            data_name
            device
            rule_format_string: str, the answer format that the solution should follow
        """

        self.model = model
        self.tokenizer = tokenizer

        self.type = type

        self.data_name = data_name 
        self.device = device
        self.rule_format_string = rule_format_string 



    def load_domain_specific_verifiers(self):
        veras = [
            "calculation_check",
            "answer_correct",
            "answer_completeness",
            "understanding_check",
        ]

        return veras
        

    def get_verifications(self, question: str, solution: str):
        '''
        Get verifications from different verifiers.

        Args:
            question: str, question
            solution: str, solution

        Returns:
            verifications: dict, verifier_name -> verifier_approval
        '''
        veras = self.load_domain_specific_verifiers()
        verifications = dict()
        for vera_type in veras:
            vera_prompt = get_vera_prompt(vera_type, question, solution)
            message = [{"role": "user", "content": vera_prompt}]
            inputs = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=4096)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            verifications[vera_type] = self.extract_verifier_approval(response)

        return verifications
    

    def extract_verifier_approval(self, verifier_response):
        '''
        Extract verifier approval from verifier response.

        Args:
            verifier_response: str, verifier response

        Returns:
            verifier_approval: bool, verifier approval
        '''
        vera_answer_symbol = VERA_ANSWER_SYMBOL.lower()
        pattern = re.compile(
            r'.*{}(.*)'.format(re.escape(vera_answer_symbol)), 
            flags=re.DOTALL | re.IGNORECASE
        )
        match = pattern.search(verifier_response)
        answer = match.group(1).strip() if match else None
        if not answer:
            print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=}, "
                        f"and full verifier_response (length {len(verifier_response)}): "
                        f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
            return False
    
        answer = answer.replace("*", "")  # Remove any asterisks (bolding)
        answer = answer.strip().lower()

        if "true" in answer:
            return True
        elif "false" in answer:
            return False
        else:
            # Check if 'true' or 'false' is in the first word
            print(colored(f"NOTICE in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                        f"checking if the FIRST WORK contains 'true' or 'false'...", "magenta"))
            first_word = answer.split()[0]
            if "true" in first_word:
                print(colored(f"\tSuccess. Found 'true' in first_word.lower(): {first_word.lower()}", "magenta"))
                return True
            elif "false" in first_word:
                print(colored(f"\tSuccess. Found 'false' in first_word.lower(): {first_word.lower()}", "magenta"))
                return False
            else:
                print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                            f"AND first word does not contain 'true' or 'false. Full verifier_response: "
                            f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
                return False
            

    def get_reward(self, question, solution):
        '''
        Get reward from question and solution.

        Args:
            question: str, question
            solution: str, solution
        Returns:
            reward: int, reward
        '''
        verifications = self.get_verifications(question, solution)
        reward = 0
        reward_list = self.get_reward_list()
        total = 0
        for verifier_name, verifier_approval in verifications.items():
            total += reward_list[verifier_name]
            if verifier_approval:
                print(colored(f"Verifier {verifier_name} approved the solution.", "green"))
            else:
                print(colored(f"Verifier {verifier_name} disapproved the solution.", "red"))
                reward -= reward_list[verifier_name]

        if self.rule_format_string is not None:
            format_approval = self.get_rule_format_verify(solution)
            if format_approval:
                print(colored(f"Verifier Rule Format approved the solution.", "green"))
            else:
                print(colored(f"Verifier Rule Format disapproved the solution.", "red"))
                reward += -2
                
        return reward / total


    def get_rule_format_verify(self, solution):
        """
        Judge whether the answer follow the format rule.

        Args:
            solution: str
        """
        answer_pattern = self.rule_format_string
        matches = list(re.finditer(answer_pattern, solution, re.DOTALL))
        if len(matches) > 0:
            return True
        else:
            return False
        
    
    def get_reward_answer_only(self, question, solution):
        '''
        Get reward based only on answer.

        Args:
            question: str, question
            solution: str, solution
        Returns:
            reward: int, reward

        Note that when using this reward function, you should only use the "answer_correct" verifier
        '''
        verifications = self.get_verifications(question, solution)
        reward = 0
        reward_list = self.get_reward_list()
        total = 0
        for verifier_name, verifier_approval in verifications.items():
            total += reward_list[verifier_name]
            if verifier_approval:
                print(colored(f"Verifier {verifier_name} approved the solution.", "green"))
            else:
                print(colored(f"Verifier {verifier_name} disapproved the solution.", "red"))
                reward -= reward_list[verifier_name]
        return reward / total
        
        
    def get_reward_list(self):
        '''
        get reward list for different verifiers
        '''
        reward_list = {
            "calculation_check": 2,
            "answer_correct": 1, 
            "answer_completeness": 2,
            "understanding_check": 1,
        }
        return reward_list

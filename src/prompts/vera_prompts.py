"""
Verification prompt from Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers.

"""
VERA_ANSWER_SYMBOL = "FINAL VERIFICATION ANSWER IS:"

def get_vera_prompt(vera_name, question, solution):
    '''
    Get prompt used for verifications.
    Args:
        vera_name: str, name of the verifier.
        question: str, the question to be verified.
        solution: str, the proposed solution to the question.
    '''
    system_str_math = (
        "You are a critical verifier tasked with evaluating mathematical problem-solving. "
        "You will be presented with a question and a proposed solution. "
        "Your job is to carefully go over and analyze the solution. Follow the instructions."
    )

    math_prefix = f"""{system_str_math}\n\n
    QUESTION:
    {question}\n\n
    PROPOSED SOLUTION:
    {solution}\n\n"""

   
    vera_names_to_prompts = {
        "calculation_check": (
            f"{math_prefix}"
            "INSTRUCTIONS:\n"
            "1. EXTRACT CALCULATION EXPRESSIONS: Extract all the mathematical calculations from the PROPOSED SOLUTION.\n"
            "2. INDEPENDENT RECOMPUTATION: Break down the calculations step-by-step and recompute them.\n"
            f"3. VERIFY: Compare your recomputation with the PROPOSED SOLUTION. If any discrepancy is found, output '{VERA_ANSWER_SYMBOL}False'. If all steps are correct, output '{VERA_ANSWER_SYMBOL}True'.\n\n"
            "NOTE: You ONLY need to check calculations(like 1 + 1 = 2, 2 * 3 = 6, etc). Ignore standalone numbers(like 1, 2, 3, etc) that are not part of a computation.\n\n"
        ),

        "answer_correct": (
            f"{math_prefix}"
            "INSTRUCTIONS:\n"
            "Your task is to determine whether the provided answer is correct.\n"
            "Think through the verification process carefully and logically.\n"
            "IMPORTANT RULES:\n"
            "1. Do NOT analyze the steps or methods used to arrive at the answer.\n"
            "2. Only evaluate the final answer's correctness.\n"
            "3. Your response must strictly follow the required format:\n"
            f"- If the answer is correct, respond with: '{VERA_ANSWER_SYMBOL}True'.\n"
            f"- If the answer is incorrect, respond with: '{VERA_ANSWER_SYMBOL}False'.\n"
        ),
        "answer_completeness": (
            f"{math_prefix}"
            "INSTRUCTIONS:\n"
            "Your task is to verify whether the solution provides a complete and final answer.\n"
            "Follow these rules carefully:\n"
            "1. Check if the solution reaches a clear and definitive final answer.\n"
            "2. The answer must not be left incomplete, such as:\n"
            "   - Ending with an unresolved expression or formula instead of a computed result.\n"
            "   - Missing a conclusion or final statement explicitly stating the final answer.\n"
            "3. If the solution is incomplete or lacks a final answer, immediately stop checking further and respond in the exact format:\n"
            f"   - '{VERA_ANSWER_SYMBOL}False'\n"
            "4. If the solution is complete and provides a final, explicit answer, respond in the exact format:\n"
            f"   - '{VERA_ANSWER_SYMBOL}True'\n"
            
            "Examples:\n"
            "Example 1:\n"
            "final answer: 8.\n"
            f"Your response: '{VERA_ANSWER_SYMBOL}True' (The solution provides a final, definitive answer of 8.)\n"

            "Example 2:\n"
            "final answer: The area of the circle is πr², where r = 4.\n"
            f"Your response: '{VERA_ANSWER_SYMBOL}False' (The answer ends with an unresolved formula, not a computed result.)\n"

            "Example 3:\n"
            "final answer: This question does not have an answer./I cannot solve this problem.\n"
            f"Your response: '{VERA_ANSWER_SYMBOL}False' (The solution lacks a clear, final answer.)\n"
        ),
       "understanding_check": (
            f"{math_prefix}"
            "INSTRUCTIONS:\n"
            "1. PROBLEM INTERPRETATION:\n"
            "   - Assess if the proposed solution clearly understands the problem statement.\n"
            "   - Ensure that the proposed solution addresses all relevant aspects of the problem, without ignoring any key detail.\n"
            "   - Flag if the solution misinterprets or overlooks the problem's core requirements or scope.\n\n"

            "2. ALIGNMENT WITH THE TASK:\n"
            "   - Verify that the solution responds to the specific question or task outlined in the problem statement.\n"
            "   - Ensure that the solution does not deviate from the problem’s context or provides an unrelated answer.\n"
            "   - Check if any critical parts of the problem have been misinterpreted or neglected.\n\n"

            "3. TERMINATION PROTOCOL:\n"
            "   - If the solution clearly misinterprets or fails to address the problem correctly, stop and respond in the exact format:\n"
            f"     - '{VERA_ANSWER_SYMBOL}False'\n"
            "   - If the solution accurately captures the problem statement and aligns with the required solution, respond in the exact format:\n"
            f"     - '{VERA_ANSWER_SYMBOL}True'\n"

            "EXAMPLES:\n"
            "[Case 1] Problem: A shop is selling a drink at 1.5 times the original price. If the original price is $10, what is the new price?\n"
            "  Solution: The new price is 1.15 * $10 = $11.50.\n"
            "  Assessment: The solution misinterprets the problem by calculating 1.15 times the original price instead of 1.5 times.\n"
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"

            "[Case 2] Problem: The second cup of coffee is half price. If the first cup costs $5, how much is the second cup?\n"
            "  Solution: The second cup costs $5 * 0.5 = $2.50.\n"
            "  Assessment: The solution correctly interprets the price as half the original price for the second cup.\n"
            f"  Result: '{VERA_ANSWER_SYMBOL}True'\n\n"

            "[Case 3] Problem: A pizza has a radius of 8 inches. What is the area of the pizza?\n"
            "  Solution: The area is π * r², where r = 4 inches. The area is 16π square inches.\n"
            "  Assessment: The solution misinterprets the formula for the area of a circle by using the radius incorrectly.\n"
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"

            "[Case 4] Problem: A train is moving at 60 km/h towards the east. What is its velocity after 2 hours?\n"
            "  Solution: The velocity is 120 km/h west.\n"
            "  Assessment: The solution correctly calculates the speed, but misinterprets the direction as west instead of east.\n"
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"

            "CRITICAL REQUIREMENTS:\n"
            "- Assess whether the solution addresses all parts of the problem.\n"
            "- Ensure the solution does not deviate from the problem’s intent.\n"
            "- Use exact output formats specified, showing no tolerance for misinterpretations."
        ),

    }
    return vera_names_to_prompts[vera_name]

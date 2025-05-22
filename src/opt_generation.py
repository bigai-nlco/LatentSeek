import torch
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
def optimized_generation(
        reward_model, model, tokenizer, device,
        question, input_text, original_answer, 
        original_hidden_states_list, input_ids, start_index=0, 
        max_num_steps=10, lr=0.03, max_new_tokens=1024,
        grad_clip=None, k=0.1, reward_threshold=-0.2):
    '''
    Generate answer using optimized generation process

    Args:
        reward_model: reward model
        model: language model
        tokenizer: tokenizer
        device: device to use
        question: question
        input_text: formatted prompt
        original_answer: original generated answer
        original_hidden_states_list: list of hidden states for each token
        input_ids: the input_ids of original generation
        start_index: the start index of the optimized hidden states
        max_num_steps: number of optimization steps
        lr: learning rate
        max_new_tokens: maximum number of new tokens to generate
        grad_clip: gradient clipping threshold
        k: ratio of update length to the total length of hidden states
        reward_threshold: threshold for the reward to stop optimization
        
    Returns:
        final_answer: the final generated answer
        reward_history: list of rewards during optimization
        original_length: length of the original answer
        optimized_length: length of the optimized answer
        update_length: length of the optimized hidden states
    '''
    eos_token = tokenizer.eos_token
    stop_words.append(eos_token)
    reward_history = []
    initial_reward = reward_model.get_reward(question, original_answer)
    
    print(f"-- Original Output: {original_answer} -- Initial Reward: {initial_reward}")
    reward_history.append(initial_reward)
    current_reward = initial_reward
    
    original_length = len(original_hidden_states_list)
    optimized_length = 0
    
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    base_input_ids = inputs.input_ids.clone()
    
    # grab update fraction
    update_length = min(int(k * original_length), 300)
    if update_length <= 0:
        print("Update Length Zero!!!")
        final_answer = original_answer
        return final_answer, reward_history, original_length, optimized_length, update_length

    optimized_hidden_states = torch.nn.Parameter(torch.stack(
        [state.clone().detach().requires_grad_(True)
        for state in original_hidden_states_list[start_index: min(start_index + update_length, len(original_hidden_states_list))]])
    )
    
    # configure optimizer
    optimizer = torch.optim.Adam([optimized_hidden_states], lr=lr)
    
    original_seq = []
    # the prompt
    original_seq.extend(input_ids[0][len(base_input_ids[-1]): len(base_input_ids[-1]) + start_index])
    
    input_ids = input_ids[:, : len(base_input_ids[-1]) + start_index]
    base_input_ids = input_ids.clone()
    new_answer = None
    
    # optimization loop
    for _ in range(max_num_steps):
        input_ids = base_input_ids.clone()
        if current_reward > reward_threshold:
            final_answer = new_answer if new_answer is not None else original_answer
            optimized_length = len(tokenizer.encode(final_answer))
            print(f"-- Final Answer: {final_answer}, -- Current Reward: {current_reward}")
            return final_answer, reward_history, original_length, optimized_length, update_length
        
        optimizer.zero_grad()
        
        logits = model.lm_head(optimized_hidden_states) #[update_length, 1, vocab_size]
        probs = torch.softmax(logits, dim=-1) + 1e-8    #[update_length, 1, vocab_size]
        
        next_token_ids = torch.argmax(probs, dim=-1)    #[update_length, 1]
        next_token_ids = next_token_ids.squeeze(-1)    #[update_length]
        log_pi_xz = torch.log(probs[torch.arange(update_length), 0, next_token_ids] + 1e-10)
        
        # total loss
        loss = - current_reward * log_pi_xz.sum()
        print(f"-- Loss: {loss.item()}")
        loss.backward(retain_graph=True)
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], grad_clip)
        optimizer.step()
        
        # update hidden states
        generated_seq = []
        generated_seq.extend(original_seq)
        with torch.no_grad():
            next_tokens = torch.argmax(model.lm_head(optimized_hidden_states), 
                                       dim=-1) #[update_length, 1]
            next_tokens = next_tokens.squeeze(-1) #[update_length]
            generated_seq.extend(next_tokens.tolist())
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(0)], dim=-1)
                
        # generate full answer
        with torch.no_grad():
            cnt = 0
            while True:
                # prompt + update fraction -> full model -> outputs
                outputs = model.model(input_ids, output_hidden_states=True)
                hidden_states = outputs[0][:, -1]
                logits = model.lm_head(hidden_states)
                next_token_id = torch.argmax(logits, dim=-1)
                new_token = tokenizer.decode(next_token_id.item())
                generated_seq.append(next_token_id.item())
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
                cnt += 1
                if new_token == eos_token:
                    break
                if cnt > max_new_tokens:
                    break
        del outputs, hidden_states, next_token_id, new_token
        del logits, next_tokens, input_ids
        torch.cuda.empty_cache()

        new_answer = tokenizer.decode(generated_seq)
        current_reward = reward_model.get_reward(question, new_answer)
        print(f"-- New Answer: {new_answer}, -- Current Reward: {current_reward}")
            
        reward_history.append(current_reward)
        
    final_answer = new_answer
    optimized_length = len(tokenizer.encode(final_answer))
    print(f"-- Final answer: {final_answer}")
    return final_answer, reward_history, original_length, optimized_length, update_length


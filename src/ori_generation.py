import torch
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
def original_generation(input_text, model, tokenizer, device):
    '''
    Generate answer using original generation process

    Args:
        input_text
        tokenizer
        device

    Returns:
        answer: original generated answer
        hidden_states_list: list of hidden states for each token
        answer_start_index: index of the hidden state where the answer begins
    '''
    eos_token = tokenizer.eos_token
    stop_words.append(eos_token)
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    new_input_ids = inputs.input_ids.clone()
    answer = []
    hidden_states_list = []
    
    cnt = 0
    while True:
        with torch.no_grad():
            outputs = model.model(new_input_ids, output_hidden_states=True)
        hidden_states = outputs[0][:, -1] # representations for last token on the last hidden layer
        hidden_states_list.append(hidden_states.clone())
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True
        
        if hidden_states.grad is not None:
            hidden_states.grad.zero_()
        
        # generate next token
        with torch.no_grad():
            eval_logits = model.lm_head(hidden_states)
            next_token_id = torch.argmax(eval_logits, dim = -1) # [1]
            new_token = tokenizer.decode(next_token_id.item())
            answer.append(next_token_id.item())
            
            if new_token in stop_words:
                break
            new_input_ids = torch.cat([new_input_ids, next_token_id.unsqueeze(0)], dim=-1)
        cnt += 1
        if cnt > 1024:
            break
    answer = tokenizer.decode(answer)
    return answer, hidden_states_list, new_input_ids


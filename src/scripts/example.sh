PATH_TO_DATA="openai/gsm8k" # path to the dataset (the path str should contain either "AIME_2024", "gsm8k", "MATH-500")
PATH_TO_MODEL="/home/plm/Qwen2.5-7B-Instruct" # path to the model 
rho=0.2 # the value of rho, which is the hyperparameter for the fractional update
lr=0.05 # the learning rate
solver_prompt_idx=1 # the index of the solver prompt to use (0 for "boxex", 1 for "json")

python main.py \
    --dataset $PATH_TO_DATA \
    --model_name_or_path $PATH_TO_MODEL \
    --output_dir ./output \
    --k $rho \
    --lr $lr \
    --solver_prompt_idx $solver_prompt_idx \
    --device "cuda" \

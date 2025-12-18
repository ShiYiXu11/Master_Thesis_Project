import torch
import gc  # Import the garbage collector module
import os
os.environ["VLLM_USE_V1"] = "1"            
os.environ["NCCL_P2P_DISABLE"] = "1"     
os.environ["NCCL_P2P_LEVEL"] = "NVL" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn" 
os.environ["HF_HOME"] = "/work/shiyxu/hf_cache"
os.environ["HF_TOKEN"] = "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"
print("Configuring vLLM environment...")
print(f"VLLM_USE_V1: {os.environ.get('VLLM_USE_V1')}")
from vllm import LLM, SamplingParams
from prompts import gen_zero_shot_prompt 
import pandas as pd
import torch.distributed as dist
import re 

def gen(model_path_A):
    model_name  = re.sub(r'^[^/]*/', '', model_path_A)
    OUTPUT_CSV_PATH_A = "gen_A_50_" + str(model_name) + "_zero_shot.csv"
    
    # Initialize Model A's engine
    # This will reserve a large block of VRAM
    llm_A = LLM(model=model_path_A, download_dir="/work/shiyxu/hf_cache"
                , quantization="awq",dtype="auto", max_model_len=1024)
    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=100,
        stop=["Input Triple:", "Input"]
    )
    
    # Load your data
    df_A = pd.read_csv("relations_A_50.csv")
    
    print("--- 2. Generating with Model A ---")
    for style in ["academic", "oral", "twitter"]:
        print(f" - Style: {style}")
        prompts_list_A = df_A.apply(
            lambda row: gen_zero_shot_prompt(input_triple=row["triples"], style=style),
            axis=1
        ).tolist()

        # Generate with Model A
        outputs_A = llm_A.generate(prompts_list_A, sampling_params)
        
        results = []
        for output in outputs_A:
            generated_text = output.outputs[0].text.strip()
            results.append(generated_text)
        df_A[style] = results
        del prompts_list_A
        del outputs_A
    
    try:
        df_A.to_csv(OUTPUT_CSV_PATH_A, index=False)
        print(f"\nModel A results saved to: {OUTPUT_CSV_PATH_A}")
    except Exception as e:
        print(f"Error saving Model A results: {e}")
    
    print("\n--- 3. Cleaning up Model A ---")
    del llm_A 
    del df_A 
    gc.collect()
    torch.cuda.empty_cache()
    
    # IMPORTANT: Destroy distributed process group if initialized
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Destroyed distributed process group")

if __name__ == "__main__":
    for model in ["casperhansen/llama-3.3-70b-instruct-awq", "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4","TheBloke/Llama-2-70B-AWQ"]:
        try:
            gen(model)
        finally:
            # Extra safety: ensure cleanup even if errors occur
            if dist.is_initialized():
                dist.destroy_process_group()

# import torch
# import gc  # Import the garbage collector module
# from vllm import LLM, SamplingParams
# # We assume you have a local 'prompts.py' with this function
# from prompts import gen_zero_shot_prompt 
# import pandas as pd
# import os
# os.environ["HF_HOME"] = "/work/shiyxu/hf_cache"
# os.environ["HF_TOKEN"] = "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"

# def gen(model_path_A):
#     OUTPUT_CSV_PATH_A = "gen_A_50_" + str(model_path_A) + "_zero_shot.csv"
    
#     # Initialize Model A's engine
#     # This will reserve a large block of VRAM
#     llm_A = LLM(model="meta-llama/"+ model_path_A, download_dir="/work/shiyxu/hf_cache",dtype="auto")
#     sampling_params = SamplingParams(
#         temperature=0.2,
#         max_tokens=100,
#         stop=["Input Triple:", "Input"]
#     )
    
#     # Load your data
#     df_A = pd.read_csv("relations_A_50.csv")
    
#     print("--- 2. Generating with Model A ---")
#     for style in ["academic", "oral", "twitter"]:
#         print(f" - Style: {style}")
#         prompts_list_A = df_A.apply(
#             lambda row: gen_zero_shot_prompt(input_triple=row["triples"], style=style),
#             axis=1
#         ).tolist()

#         # Generate with Model A
#         outputs_A = llm_A.generate(prompts_list_A, sampling_params)
        
#         results = []
#         for output in outputs_A:
#             generated_text = output.outputs[0].text.strip()
#             results.append(generated_text)
#         df_A[style] = results
#         del prompts_list_A
#         del outputs_A
    
#     try:
#         df_A.to_csv(OUTPUT_CSV_PATH_A, index=False)
#         print(f"\nModel A results saved to: {OUTPUT_CSV_PATH_A}")
#     except Exception as e:
#         print(f"Error saving Model A results: {e}")
#     print("\n--- 3. Cleaning up Model A ---")
#     del llm_A 
#     del df_A 
#     gc.collect()
#     torch.cuda.empty_cache()
# for model in ["Llama-2-7b-hf","Llama-2-7b-chat-hf","Llama-2-13b-hf","Llama-2-13b-chat-hf","Llama-3.1-8B","Llama-3.1-8B-Instruct","Llama-3.2-1B","Llama-3.2-3B","Llama-3.2-1B-Instruct","Llama-3.2-3B-Instruct"]:
#     gen(model)


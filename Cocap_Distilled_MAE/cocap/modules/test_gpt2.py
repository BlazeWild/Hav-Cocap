import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_gpt2_standalone():
    print("🚀 Initializing GPT-2 Standalone Test...")
    
    # Path to your offline folder from url.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "../../model_zoo/gpt2_model")
    
    try:
        # 1. Load Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 2. Load Model (Force to CPU for easy testing)
        model = GPT2LMHeadModel.from_pretrained(model_dir).to("cpu")
        model.eval()
        
        print("✅ Model and Tokenizer loaded successfully from offline folder.")

        # 3. The Prompt
        prompt = "Hello, I'm a language model"
        inputs = tokenizer(prompt, return_tensors="pt")

        print(f"📝 Prompting: '{prompt}'")
        
        # 4. Generate
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )

        # 5. Result
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        print("\n" + "="*40)
        print("GPT-2 SAYS:")
        print(result)
        print("="*40)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nCheck if your 'gpt2_model' folder has: config.json, model.safetensors, vocab.json, and merges.txt")

if __name__ == "__main__":
    test_gpt2_standalone()
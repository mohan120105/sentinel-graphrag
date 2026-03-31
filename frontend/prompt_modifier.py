from llama_cpp import Llama
import time

# 1. Load the 4-bit quantized model into memory (do this once at startup)
# Adjust n_threads to match your CPU's physical cores for max speed
modifier_llm = Llama(
    model_path="./models/gemma-1b-it-Q4_K_M.gguf", # Update with your exact filename
    n_ctx=256,   # Small context window for speed
    n_threads=2,  # Uses 4 CPU cores
    verbose=False # Hides the C++ loading logs in your terminal
)

def enhance_query_for_graphrag(user_query: str) -> str:
    """Passes the raw query to the 4-bit model to extract intent and format it."""
    
    # Using Gemma's specific prompt template formatting
    prompt = (
        "<bos><start_of_turn>user\n"
        "You are an expert banking AI assistant. Rewrite the following user input into a highly specific, professional query optimized for a GraphRAG database search. "
        "Keep it under 2 sentences. Do not add conversational filler. \n\n"
        f"User Input: '{user_query}'<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    start_time = time.time()
    
    response = modifier_llm(
        prompt,
        max_tokens=60, # Keep generation short for speed
        stop=["<end_of_turn>", "\n\n"], 
        echo=False
    )
    
    enhanced_query = response['choices'][0]['text'].strip()
    execution_time = round(time.time() - start_time, 2)
    
    print(f"⚡ Modifier ran in {execution_time}s")
    return enhanced_query

# --- Quick Local Test ---
if __name__ == "__main__":
    raw_input = "nri docs needed"
    print(f"Raw Input: {raw_input}")
    print(f"Enhanced:  {enhance_query_for_graphrag(raw_input)}")
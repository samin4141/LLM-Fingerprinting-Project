"""
Test script for Mistral 7B model using Ollama
"""

import ollama

def test_mistral(prompt: str, temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9, 
                 repeat_penalty: float = 1.1, num_predict: int = None):
    """
    Test the Mistral 7B model with a given prompt.
    
    Args:
        prompt: The input prompt/question to send to the model
        temperature: Controls randomness (0.0 = deterministic, higher = more creative). Default: 0.7
        top_k: Limits sampling to top K tokens. Default: 40
        top_p: Nucleus sampling - considers tokens with cumulative probability up to top_p. Default: 0.9
        repeat_penalty: Penalty for repetition (1.0 = no penalty, >1.0 = less repetition). Default: 1.1
        num_predict: Maximum number of tokens to generate. Default: None (no limit)
    """
    print(f"Prompt: {prompt}\n")
    print(f"Parameters: temperature={temperature}, top_k={top_k}, top_p={top_p}, repeat_penalty={repeat_penalty}")
    print("Response from Mistral 7B:")
    print("-" * 50)
    
    try:
        # Build options dictionary with decoding parameters
        options = {
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'repeat_penalty': repeat_penalty,
        }
        if num_predict is not None:
            options['num_predict'] = num_predict
        
        # Generate response using Mistral 7B
        response = ollama.generate(
            model='mistral',
            prompt=prompt,
            options=options
        )
        
        # Print the response
        print(response['response'])
        print("-" * 50)
        
        # Print some metadata
        if 'context' in response:
            print(f"\nContext tokens: {len(response.get('context', []))}")
        if 'total_duration' in response:
            print(f"Total duration: {response['total_duration'] / 1e9:.2f} seconds")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running. You can start it by running 'ollama serve' in a terminal.")


def test_chat_mode(temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9, 
                   repeat_penalty: float = 1.1, num_predict: int = None):
    """
    Test Mistral 7B in chat mode (conversational).
    
    Args:
        temperature: Controls randomness (0.0 = deterministic, higher = more creative). Default: 0.7
        top_k: Limits sampling to top K tokens. Default: 40
        top_p: Nucleus sampling - considers tokens with cumulative probability up to top_p. Default: 0.9
        repeat_penalty: Penalty for repetition (1.0 = no penalty, >1.0 = less repetition). Default: 1.1
        num_predict: Maximum number of tokens to generate. Default: None (no limit)
    """
    print("\n" + "=" * 50)
    print("Testing Chat Mode")
    print("=" * 50 + "\n")
    
    messages = [
        {
            'role': 'user',
            'content': 'What is machine learning? Explain it in simple terms.'
        }
    ]
    
    print(f"User: {messages[0]['content']}\n")
    print(f"Parameters: temperature={temperature}, top_k={top_k}, top_p={top_p}, repeat_penalty={repeat_penalty}")
    print("Mistral 7B:")
    print("-" * 50)
    
    try:
        # Build options dictionary with decoding parameters
        options = {
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'repeat_penalty': repeat_penalty,
        }
        if num_predict is not None:
            options['num_predict'] = num_predict
        
        response = ollama.chat(
            model='mistral',
            messages=messages,
            options=options
        )
        
        print(response['message']['content'])
        print("-" * 50)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Test 1: Simple generation with default parameters
    print("=" * 50)
    print("Test 1: Simple Prompt Generation (Default Parameters)")
    print("=" * 50 + "\n")
    
    test_prompt = "Write a short Python function to calculate the factorial of a number."
    test_mistral(test_prompt)
    
    # Test 2: High temperature (more creative/random)
    print("\n" + "=" * 50)
    print("Test 2: High Temperature (More Creative)")
    print("=" * 50 + "\n")
    
    test_mistral(test_prompt, temperature=1.2, top_k=50)
    
    # Test 3: Low temperature (more deterministic/focused)
    print("\n" + "=" * 50)
    print("Test 3: Low Temperature (More Deterministic)")
    print("=" * 50 + "\n")
    
    test_mistral(test_prompt, temperature=0.3, top_k=20)
    
    # Test 4: Chat mode with custom parameters
    test_chat_mode(temperature=0.8, top_k=30)
    
    # Test 5: Creative writing with high temperature
    print("\n" + "=" * 50)
    print("Test 5: Creative Writing (High Temperature)")
    print("=" * 50 + "\n")
    
    creative_prompt = "Write a haiku about artificial intelligence."
    test_mistral(creative_prompt, temperature=1.0, top_k=50, repeat_penalty=1.15)


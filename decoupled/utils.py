import os

def load_prompt(file_name: str) -> str:
    """
    Loads a prompt from a .txt file in the root 'prompts' directory.
    """
    try:
        # Get the directory of the current script (utils.py)
        script_dir = os.path.dirname(__file__)
        
        # Go up one level to the project root (from 'decoupled' to root)
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        
        # Construct the full path to the prompt file
        prompt_path = os.path.join(project_root, "prompts", file_name)
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}")
        print("Please make sure the 'prompts' directory and its files exist in the project root.")
        raise
    except Exception as e:
        print(f"Error loading prompt {file_name}: {e}")
        raise
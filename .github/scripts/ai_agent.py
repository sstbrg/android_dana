import os
import sys
import requests
import re
from pathlib import Path

# --- Configuration ---
# Get the API key and issue details from environment variables set by the GitHub Action
API_KEY = os.getenv("AI_API_KEY")
ISSUE_TITLE = os.getenv("ISSUE_TITLE")
ISSUE_BODY = os.getenv("ISSUE_BODY")
LLM_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
FILE_SEPARATOR = "---FILE_SEPARATOR---"

def call_llm(prompt: str) -> str:
    """
    Calls the configured LLM API with a given prompt and handles potential errors.

    Args:
        prompt: The prompt to send to the language model.

    Returns:
        The text content from the LLM's response.
        
    Raises:
        requests.exceptions.RequestException: If the API call fails.
        KeyError: If the response JSON is not in the expected format.
        ValueError: If no valid response candidate is found.
    """
    if not API_KEY:
        print("Error: AI_API_KEY environment variable not set.")
        sys.exit(1)

    print("Sending prompt to LLM...")
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    params = {"key": API_KEY}

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload, params=params, timeout=300)
        response.raise_for_status()  # Fail if the request was unsuccessful (e.g., 4xx or 5xx error)
        
        response_data = response.json()
        
        # Safely navigate the response JSON
        candidates = response_data.get('candidates', [])
        if not candidates:
            raise ValueError("No candidates found in LLM response.")
            
        content = candidates[0].get('content', {})
        parts = content.get('parts', [])
        if not parts:
            raise ValueError("No parts found in LLM response content.")

        print("LLM response received successfully.")
        return parts[0].get('text', '')

    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        # Log the response body if available for more context on the error
        if e.response is not None:
            print(f"Response Body: {e.response.text}")
        sys.exit(1)
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Full Response: {response_data}")
        sys.exit(1)


def parse_and_write_files(response_text: str):
    """
    Parses the LLM response text, which contains multiple files,
    and writes each file to its specified path.

    Args:
        response_text: The raw text from the LLM containing file blocks.
    """
    print("Parsing response and writing files...")
    # Split the response into individual file blocks
    file_blocks = response_text.split(FILE_SEPARATOR)

    for block in file_blocks:
        if not block.strip():
            continue

        # Use regex to find the file path comment
        match = re.search(r"^\s*#\s*FILE_PATH:\s*(.*)", block, re.MULTILINE)
        if not match:
            print(f"Warning: Could not find FILE_PATH in block:\n---\n{block[:200]}...\n---")
            continue

        file_path_str = match.group(1).strip()
        
        # Clean up potential markdown code fences
        content_start_index = match.end()
        clean_block = block[content_start_index:].strip()
        if clean_block.startswith("```"):
            # Find the end of the first line to remove the language specifier (e.g., ```kotlin)
            first_line_end = clean_block.find('\n') + 1
            clean_block = clean_block[first_line_end:]
        if clean_block.endswith("```"):
            clean_block = clean_block[:-3]
        
        content = clean_block.strip()

        try:
            file_path = Path(file_path_str)
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the content to the file
            file_path.write_text(content, encoding='utf-8')
            print(f"Successfully wrote to {file_path}")

        except Exception as e:
            print(f"Error writing file {file_path_str}: {e}")
            sys.exit(1)


def main():
    """
    Main function to orchestrate the AI agent's tasks.
    """
    if not ISSUE_TITLE or not ISSUE_BODY:
        print("Error: ISSUE_TITLE or ISSUE_BODY environment variables not set.")
        sys.exit(1)

    # --- Step 1: Generate Application Code ---
    print("\n--- Step 1: Generating Application Code ---")
    code_gen_prompt = f"""
    You are an expert Android developer specializing in Kotlin and modern Android practices.
    Your task is to generate all necessary files for a new feature based on a GitHub issue.

    GitHub Issue Title: '{ISSUE_TITLE}'
    GitHub Issue Body:
    ---
    {ISSUE_BODY}
    ---

    Instructions:
    1.  Generate complete, production-ready Kotlin and XML layout files.
    2.  Ensure the code is clean, well-commented, and follows standard Android architecture patterns (e.g., MVVM if applicable).
    3.  IMPORTANT: Your response MUST be a single block of text. Separate each file's content with the exact delimiter: '{FILE_SEPARATOR}'.
    4.  At the beginning of each file's content, you MUST include a comment indicating its full, relative path from the project root. The format is critical. Example:
        # FILE_PATH: app/src/main/java/com/example/myapp/MyNewActivity.kt
    5.  For XML files, use a comment like this:
        <!-- FILE_PATH: app/src/main/res/layout/activity_my_new.xml -->
    """
    
    generated_code_response = call_llm(code_gen_prompt)
    parse_and_write_files(generated_code_response)
    print("Application code generation complete.")

    # --- Step 2: Generate Unit Tests ---
    print("\n--- Step 2: Generating Unit Tests ---")
    test_gen_prompt = f"""
    You are an expert Android test engineer. Your task is to write comprehensive unit tests for the provided application code.

    Application Code to Test:
    ---
    {generated_code_response}
    ---

    Instructions:
    1.  Use JUnit 5 and Mockito for testing.
    2.  Generate a complete test file that covers the logic in the provided code. Include tests for happy paths and edge cases.
    3.  As before, your response must be a single block of text, and you MUST start the file content with a file path comment. Example:
        # FILE_PATH: app/src/test/java/com/example/myapp/MyNewActivityTest.kt
    """
    
    generated_test_response = call_llm(test_gen_prompt)
    parse_and_write_files(generated_test_response)
    print("Unit test generation complete.")
    print("\nAI Agent finished successfully.")


if __name__ == "__main__":
    main()

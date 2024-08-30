import os
import subprocess

# Define the path to your text file
TEXT_FILE_PATH = '/home/dq/Desktop/demo/pdftojson/output/Allen Dunfee Face Sheet.txt'  # Update this path if needed

def read_text_file(file_path):
    """Read the content of the text file."""
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_response_from_ollama(prompt):
    """Send the prompt to the local Ollama model and get the response."""
    try:
        # Construct the command to run Ollama with the prompt
        # Replace <model_name> with the actual model name or identifier
        command = f'ollama run gemma2 "{prompt}"'  # Update this based on your findings

        # Execute the command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            print(f"Error running Ollama: {result.stderr}")
            return None
        
        return result.stdout

    except Exception as e:
        print(f"Error during request: {e}")
        return None

def main():
    """Main function to read text, send it to Ollama, and print the response."""
    text = read_text_file(TEXT_FILE_PATH)
    if text:
        print("Original Text:")
        text = text + "\n Generate JSon File for this data donot miss anything icnlude null values too"
        print(text)
        print("\nGenerating response...\n")
        response = get_response_from_ollama(text)
        if response:
            print("Generated Response:")
            print(response.strip())  # Print the response without extra newlines
        else:
            print("No response generated.")
    else:
        print("No text to process.")

if __name__ == '__main__':
    main()

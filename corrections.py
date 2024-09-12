from spellchecker import SpellChecker

def correct_spelling_in_file(input_file, output_file):
    # Initialize the spell checker
    spell = SpellChecker()

    # Read the original file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Create an output file to save corrected content
    with open(output_file, 'w') as corrected_file:
        for line in lines:
            corrected_line = []
            words = line.split()  # Split by whitespace to keep track of layout
            
            for word in words:
                # Ignore special characters and numbers
                cleaned_word = ''.join([char for char in word if char.isalpha()])
                
                # Correct word only if it's not a special character/number
                if cleaned_word:
                    corrected_word = spell.correction(cleaned_word)
                    
                    # If corrected_word is None, use the original cleaned_word
                    if corrected_word is None:
                        corrected_word = cleaned_word

                    corrected_line.append(word.replace(cleaned_word, corrected_word))
                else:
                    corrected_line.append(word)
                    
            # Join the corrected words back into the original line layout
            corrected_file.write(' '.join(corrected_line) + '\n')

input_file = "/home/poorna/Desktop/pdf2json/OCR/output/Allen Dunfee Face Sheet.txt"
output_file = "/home/poorna/Desktop/pdf2json/OCR/output/Corrected_Allen_Dunfee_Face_Sheet.txt"
correct_spelling_in_file(input_file, output_file)

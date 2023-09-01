from transformers import MarianMTModel, MarianTokenizer
import torch

# Load the pre-trained model and tokenizer for English to Hindi translation
model_name = "Helsinki-NLP/opus-mt-en-hi"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate_to_hinglish(text, model, tokenizer):
    # Tokenize the input text and convert it to IDs
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    # Generate translation using the model
    with torch.no_grad():
        translated_ids = model.generate(input_ids)
    
    # Convert the translated IDs back to text
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    
    return translated_text

english_text = input("enter your text:")
translated_text = translate_to_hinglish(english_text, model, tokenizer)
print(translated_text)

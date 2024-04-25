import csv
import torch
from network import Network
from vocab import build_vocab
from preprocess import parse_patient
from utils import evaluate_ddx
from questionare import process_evidences
from transformers import MarianMTModel, MarianTokenizer

# Model and tokenizer for French to English translation
model_name_en_fr = "Helsinki-NLP/opus-mt-en-fr"
tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr)
model_en_fr = MarianMTModel.from_pretrained(model_name_en_fr)

# Model and tokenizer for English to Hindi translation (optional)
model_name_en_hi = "Helsinki-NLP/opus-mt-en-hi"  # Replace with your desired English-Hindi model
tokenizer_en_hi = MarianTokenizer.from_pretrained(model_name_en_hi)
model_en_hi = MarianMTModel.from_pretrained(model_name_en_hi)

def translate_to_french(text):
  tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr)  # Define tokenizer_en_fr here
  inputs = tokenizer_en_fr(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
  translated_ids = model_en_fr.generate(**inputs)
  translated_text = tokenizer_en_fr.decode(translated_ids[0], skip_special_tokens=True)
  return translated_text

def translate_to_english(text):
    """
    Translates text from French to English using the MarianMT model.

    Args:
        text: The text to be translated (French).

    Returns:
        The translated text in English.
    """
    # Define the model name for the French to English translation
    model_name_fr_en = "Helsinki-NLP/opus-mt-fr-en"

    # Load the tokenizer for French-English translation
    tokenizer_fr_en = MarianTokenizer.from_pretrained(model_name_fr_en)

    # Tokenize the input text and generate translations
    inputs = tokenizer_fr_en(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_ids = model_en_fr.generate(**inputs)

    # Decode the translated text and remove special tokens
    translated_text = tokenizer_fr_en.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

def translate_to_hindi(text):
  inputs = tokenizer_en_hi(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
  translated_ids = model_en_hi.generate(**inputs)
  translated_text = tokenizer_en_hi.decode(translated_ids[0], skip_special_tokens=True)
  return translated_text
evidences = process_evidences()
# Translate the input parameters to French
en_input = {
    'AGE': 2,
    'PATHOLOGY': "NONE",
    'SEX': 'M',
    'EVIDENCES': evidences,
    'DIFFERENTIAL_DIAGNOSIS': [                                           
        ["Exacerbated asthma or bronchospasm", 0.0802201660064834],
        ["Possible influenza or typical viral syndrome", 0.06940463222398906],
        ["Viral pharyngitis", 0.0655942933919721],
        ["Allergic rhinitis", 0.06552892428208987],
        ["Pneumonia", 0.06545017163298406],
        ["Bronchitis", 0.06364201476444997],
        ["Spontaneous pneumothorax", 0.059378243534228575],
        ["Tuberculosis", 0.04690915875550665],
        ["RSV or viremia", 0.046759599824355666],
        ["Myocarditis", 0.04651861210731325],
        ["Anaphylaxis", 0.04572069791718869],
        ["Acute laryngitis", 0.04358843624121155],
        ["Guillain-Barré syndrome", 0.04179138110732587],
        ["Laryngotracheobronchitis (Croup)", 0.04179138110732587],
        ["Atrial fibrillation/Flutter auricular", 0.03991832618321649],
        ["Acute dystonic reaction", 0.02942992438565071],
        ["Grave's disease", 0.02942992438565071],
        ["Anemia", 0.02942992438565071],
        ["Scombroid", 0.027190751855561253],
        ["Sarcoidosis", 0.024100679101770887],
        ["Venous thromboembolism (VTE)", 0.015720427742588923],
        ["Systemic lupus erythematosus (SLE)", 0.013624650147256443],
        ["Chagas disease", 0.008857678916229164]
    ],
    'INITIAL_EVIDENCE': "painxx"
}

translated_input = {}
for key, value in en_input.items():
    if isinstance(value, str):  # Translate strings only
        translated_input[key] = translate_to_french(value)
    else:
        translated_input[key] = value

# Load the trained network
batch_size = 64
vocab_size = 436
en_seq_len = 80
de_seq_len = 40
features = 128
heads = 4
layers = 6
output_size = 54
drop_rate = 0.1

network = Network(vocab_size=vocab_size,
                  en_seq_len=en_seq_len,
                  de_seq_len=de_seq_len,
                  features=features,
                  heads=heads,
                  n_layer=layers,
                  output_size=output_size,
                  dropout_rate=drop_rate).cpu()

network.load_state_dict(torch.load('model.h5', map_location=torch.device('cpu')))

# loading inference sample
en_vocab, de_vocab = build_vocab()

en_input, de, gt = parse_patient(translated_input, en_max_len=80, de_max_len=41)

en_input = list(map(lambda x: en_vocab.get(x, en_vocab['<unk>']), en_input.split(' ')))
de = list(map(lambda x: de_vocab.get(x, de_vocab['<unk>']), de.split(' ')))
de_input = de[0:-1]
de_output = de[1:]
pathology = de_vocab.get(gt)

# inference
network.eval()
with torch.no_grad():
    en_input = torch.tensor([en_input]).long().cpu()
    de_input = torch.tensor([de_input]).long().cpu()
    de_output = torch.tensor([de_output]).long().cpu()
    de_in_ = torch.zeros((1, 40)).long().cpu()
    de_in_[:, 0] = 1  # start decoder with <bos> token

    for i in range(1):
        y_pred, cls_ = network(en_input=en_input, de_input=de_in_)
        p_ = torch.argmax(y_pred, dim=-1)
        de_in_[:, i + 1] = p_[:, i]
        if p_[:, i] == 0:
            break

    # Convert tensor index to pathology code
    
predicted_pathology_index = torch.argmax(cls_, dim=-1).item()
    # Get the predicted pathology code in French
predicted_pathology_code_fr = list(de_vocab.keys())[list(de_vocab.values()).index(predicted_pathology_index)]

# Translate the predicted pathology code from French to English
predicted_pathology_code_en = translate_to_english(predicted_pathology_code_fr)
predicted_pathology_code_hi = translate_to_hindi(predicted_pathology_code_en)
print('Predicted Pathology (French):', predicted_pathology_code_fr)
print('Predicted Pathology (English):', predicted_pathology_code_en)
print('Predicted Pathology (hindi):', predicted_pathology_code_hi)


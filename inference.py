import csv
import torch
from network import Network
from vocab import build_vocab
from preprocess import parse_patient
from utils import evaluate_ddx

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

# filename = 'data/release_test_patients.csv'
# with open(filename, mode='r', encoding='utf-8') as f:
#     loader = list(csv.DictReader(f))

import csv
import torch
from network import Network
from vocab import build_vocab
from preprocess import parse_patient
from utils import evaluate_ddx
from transformers import MarianMTModel, MarianTokenizer

# Load the pre-trained translation model and tokenizer for English to French
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_to_french(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_ids = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

# Translate the input parameters to French
en_input = {
    'AGE': 2,
    'PATHOLOGY': "Bronchitis",
    'SEX': 'M',
    'EVIDENCES': [
        "painxx",
        "painxx_char_@_a_burn_or_heat",
        "painxx_bodypart_@_side_of_the_thorax_D_",
        "painxx_bodypart_@_pharynx",
        "painxx_intensity_@_5",
        "painxx_irrad_@_none",
        "painxx_precis_@_7",
        "painxx_sudden_@_4",
        "dyspnea",
        "j44_j42",
        "clear_rhino",
        "cough",
        "trav1_@_N",
        "wheezing"
    ],
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

print("Translated Input:")
print(en_input)
print(de)
print(gt)

en_input = list(map(lambda x: en_vocab.get(x, en_vocab['<unk>']), en_input.split(' ')))
de = list(map(lambda x: de_vocab.get(x, de_vocab['<unk>']), de.split(' ')))
de_input = de[0:-1]
de_output = de[1:]
pathology = de_vocab.get(gt)

print("\nGround truth encoder input: ", en_input)
print("Ground truth decoder input: ", de_input)
print("Ground truth decoder output: ", de_output)
print("Ground truth pathology: ", pathology)

# Inference
network.eval()
with torch.no_grad():
    en_input = torch.tensor([en_input]).long().cpu()
    de_input = torch.tensor([de_input]).long().cpu()
    de_output = torch.tensor([de_output]).long().cpu()
    de_in_ = torch.zeros((1, 40)).long().cpu()
    de_in_[:, 0] = 1  # start decoder with <bos> token

    for i in range(40 - 1):
        print(i, de_in_.tolist())
        y_pred, cls_ = network(en_input=en_input, de_input=de_in_)
        p_ = torch.argmax(y_pred, dim=-1)
        de_in_[:, i + 1] = p_[:, i]
        if p_[:, i] == 0:
            break

    acc = evaluate_ddx(true=de_output, pred=y_pred)
    print('Accuracy:', acc)
    print('Predicted Pathology:', torch.argmax(cls_, dim=-1))


print(en_input)
print(de)
print(gt)

en_input = list(map(lambda x: en_vocab.get(x, en_vocab['<unk>']), en_input.split(' ')))
de = list(map(lambda x: de_vocab.get(x, de_vocab['<unk>']), de.split(' ')))
de_input = de[0:-1]
de_output = de[1:]
pathology = de_vocab.get(gt)
print(f'Ground truth encoder input: {en_input}')
print(f'Ground truth decoder input: {de_input}')
print(f'Ground truth decoder output: {de_output}')
print(f'Ground truth pathology: {pathology}')

# inference
network.eval()
with torch.no_grad():
    en_input = torch.tensor([en_input]).long().cpu()
    de_input = torch.tensor([de_input]).long().cpu()
    de_output = torch.tensor([de_output]).long().cpu()
    de_in_ = torch.zeros((1, 40)).long().cpu()
    de_in_[:, 0] = 1  # start decoder with <bos> token
    # out = None

    for i in range(40 - 1):
        print(i, de_in_.tolist())
        y_pred, cls_ = network(en_input=en_input, de_input=de_in_)
        p_ = torch.argmax(y_pred, dim=-1)
        de_in_[:, i + 1] = p_[:, i]
        if p_[:, i] == 0:
            break

            # if torch.eq():
    acc = evaluate_ddx(true=de_output, pred=y_pred)
    print('accuracy', acc)
    print(torch.argmax(cls_, dim=-1))
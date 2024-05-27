import pandas as pd

# File paths
source_file = 'test/test.nl'
reference_file = 'test/test.en'
word_model_file = 'transformer_word_2k/00013500.hyps.test'
bpe2000_model_file = 'models/bpe2k/00040500.hyps.test' 
bpe4000_model_file = 'models/bpe4k/00020000.hyps.test'

# Load the data
def load_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

source_sentences = load_file(source_file)
reference_sentences = load_file(reference_file)
word_model_sentences = load_file(word_model_file)
bpe2000_model_sentences = load_file(bpe2000_model_file)
bpe4000_model_sentences = load_file(bpe4000_model_file)

# Create a DataFrame for manual evaluation
data = {
    'Source (Dutch)': source_sentences,
    'Reference (English)': reference_sentences,
    'Word-Level Translation': word_model_sentences,
    'BPE 2000 Translation': bpe2000_model_sentences,
    'BPE 4000 Translation': bpe4000_model_sentences
}

df = pd.DataFrame(data)
df_sample = df.sample(100)  # Select a sample of 100 sentences for manual evaluation

# Save to CSV for manual evaluation
df_sample.to_csv('manual_evaluation_sample.csv', index=False)
print("Sample saved to 'manual_evaluation_sample.csv' for manual evaluation.")
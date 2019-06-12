import pandas as pd
from lda2vec.nlppipe import Preprocessor

# Data directory
data_dir ="tests//twenty_newsgroups//data"
# Where to save preprocessed data
clean_data_dir = "tests//twenty_newsgroups//data//clean_data"
# Name of input file. Should be inside of data_dir
input_file = "20_newsgroups.txt"
# Should we load pretrained embeddings from file
load_embeds = True

# Read in data file
df = pd.read_csv(data_dir+"//"+input_file, sep="\t")
df = df[:500]

# Initialize a preprocessor
P = Preprocessor(df, "texts", token_type="lower", max_features=10000,
                 maxlen=10000, min_count=3, nlp="en_core_web_lg")

# Run the preprocessing on your dataframe
P.preprocess()

# Load embeddings from file if we choose to do so
if load_embeds:
    # Load embedding matrix from file path - change path to where you saved them
    embedding_matrix = P.load_glove(300, "glove.6B.300d.txt")
else:
    embedding_matrix = None

# Save data to data_dir
P.save_data(clean_data_dir, embedding_matrix=embedding_matrix)

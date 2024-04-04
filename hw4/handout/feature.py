import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def get_embedding(dic,data):
    total = []
    for x in data:
        label = x[0]
        s = x[1].split()
        num = 0
        acc = np.zeros(300)
        for i in s:
            if i in dic:
                num+=1
                acc += dic[i]
        acc = (1/num)*acc
        acc = np.insert(acc,0,round(label,6))
        result = [round(num,6) for num in acc]
        total.append(result)
    return total

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()
    train_input = args.train_input
    test_input = args.test_input
    val_input = args.validation_input
    train_out = args.train_out
    test_out = args.test_out
    val_out = args.validation_out
    dic = load_feature_dictionary(args.feature_dictionary_in)

    train_data = load_tsv_dataset(train_input)
    test_data = load_tsv_dataset(test_input)
    val_data = load_tsv_dataset(val_input)

    train_glove = get_embedding(dic,train_data)
    test_glove = get_embedding(dic,test_data)
    val_glove = get_embedding(dic,val_data)
    print(val_glove)

    train_string = ""
    for x in train_glove:
        for i in x:
            train_string += f"{i}"+" "
        train_string += "\n"
    train_string = train_string[0:-1]

    with open(train_out,'w') as file:
        file.write(train_string)

    val_string = ""
    for x in val_glove:
        for i in x:
            val_string += f"{i}"+" "
        val_string += "\n"
    val_string = val_string[0:-1]
    with open(val_out,'w') as file:
        file.write(val_string)
    
    test_string = ""
    for x in test_glove:
        for i in x:
            test_string += f"{i}"+" "
        test_string += "\n"
    test_string = test_string[0:-1]
    with open(test_out,'w') as file:
        file.write(test_string)


    



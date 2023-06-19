import os
import ast


def create_vocab(path):
    textfiles = os.listdir(path)
    vocab = set()

    for file in textfiles:

        with open(f"{path}/{file}", "r") as f:
            print(f"Reading file {file}")
            word_dict = ast.literal_eval(f.readlines()[0])

            for word in word_dict['words']:
                vocab.update(word.lower())

    return sorted(vocab)


print("".join(create_vocab("C:\\Users\\Kare4U\\Downloads\\augmented_FUNSD\\augmented_FUNSD_texts")))

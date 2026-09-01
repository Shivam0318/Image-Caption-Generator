import string
import os
Captions_file='data/captions.txt'
def load_doc(filename):
    try:
        with open(filename, 'r', encoding= 'utf-8') as file:
            text= file.read()
            return text
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")       
        return None
def load_descriptions(doc):
    """
    Parses the document and maps image names to their 5 captions.
    Expected format: image_name.jpg,Caption text here
    """
    mapping=dict()
    for line in doc.split('\n'):
        if len(line)<2:
            continue
        tokens=line.split(',')
        if len(tokens)<2:
            continue
        image_id= tokens[0]
        image_id=image_id.split('.')[0]
        image_desc= ' '.join(tokens[1:])
        if image_id not in mapping:
            mapping[image_id]=list()
        mapping[image_id].append(image_desc)
    return mapping
def clean_descriptions(mapping):
    """
    Cleans the descriptions by converting to lowercase, removing punctuation, and filtering out short words.
    """
    table=str.maketrans('','', string.punctuation)
    for image_id,desc_list in mapping.items():
        for i in range(len(desc_list)):
            desc=desc_list[i]
            desc=desc.split()
            desc=[word.lower() for word in desc]
            desc=[word.translate(table) for word in desc]
            desc=[word for word in desc if len(word)>1]
            desc=[word for word in desc if word.isalpha()]
            desc='startseq ' + ' '.join(desc) + ' endseq'
            desc_list[i]=desc
    return mapping
if __name__ == "__main__":
    print("1. Loadiing raw text document....")
    doc=load_doc(Captions_file)
    if doc is not None:
        print("2. Mapping caption to image....")
        descriptions=load_descriptions(doc)
        print(f"Loaded:{len(descriptions)} unique images")

        print("3. Cleaning text and adding tokens....")
        clean_descriptions=clean_descriptions(descriptions)

        sample_key= list(clean_descriptions.keys())[0]
        print(f"\nSuccess! Here is a sample of the cleaned data for image'{sample_key}':")
        for caption in clean_descriptions[sample_key]:
            print(f"-{caption}")
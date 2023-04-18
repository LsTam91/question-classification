import random
import nltk
from textattack.augmentation.recipes import EasyDataAugmenter
from typing import List, Dict, Any, Union

def swap_2(d1: Dict[str, Any], d2: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Applies a noise augmentation by swapping questions between two given data
    points and concatenating them.

    Args:
        d1: The first data point containing 'question' and 'text' keys.
        d2: The second data point containing 'question' and 'text' keys.

    Returns:
        A list of dictionaries with 'input' and 'target' keys.
        'target' is 0 when the question is not answerable with the given context
    """
    new_data = []
    
    context = d1['question'] + ' </s> ' + d2['text']
    new_data.append({'input': context, 'target': 0})
    
    context = d2['question'] + ' </s> ' + d1['text']
    new_data.append({'input': context, 'target': 0})

    return new_data

def crop_words(question: str) -> str:
    """
    Applies a crop words augmentation by removing a random subset of words from a
    given question.

    Args:
        question: A string containing the question to be cropped.

    Returns:
        A string containing the cropped question.
    """
    words = question.split()
    if len(words) < 4:
        return words[-1]
    num_words = random.randint(2, len(words) // 2 + 1)
    indexes = random.sample(range(len(words)), len(words) - num_words)
    indexes.sort()
    cropped_words = [words[i] for i in indexes]
    return ' '.join(cropped_words)

def remove_subjects(question: str) -> str:
    """
    Applies a remove subjects augmentation by removing a random subset of
    nouns or pronouns from a given question.

    Args:
        question: A string containing the question to remove subjects from.

    Returns:
        A string containing the question without subjects.
    """
    # Tokenize the question
    tokens = nltk.word_tokenize(question)

    # Part of speech tag the tokens
    pos_tags = nltk.pos_tag(tokens)

    # Find the indices of nouns or pronouns in the question
    subject_indices = []
    for i, (token, pos) in enumerate(pos_tags):
        if pos in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
            subject_indices.append(i)
    
    # Choose between 1 to 3 of them to remove
    num_words = random.randint(1, len(subject_indices) // 2 + 1)
    
    # Security to not have an error if no nouns or pronouns are found
    if subject_indices == []:
        return ' ?'

    to_crop = random.sample(subject_indices, num_words)

    # Remove the subjects from the question
    question_without_subjects = " ".join([t for i, t in enumerate(tokens) if i not in to_crop])

    return question_without_subjects

def noun_question(text: str) -> str:
    """
    Applies a noun question augmentation by randomly selecting 1 to 5
    nouns or pronouns from a given text and forming a new question
    with them.

    Args:
        text: A string containing the text to select nouns from.

    Returns:
        A string containing the new question with the selected nouns.
    """
    # Tokenize the question
    tokens = nltk.word_tokenize(text)

    # Part of speech tag the tokens
    pos_tags = nltk.pos_tag(tokens)
    
    subject_indices = []
    for i, (token, pos) in enumerate(pos_tags):
        if pos in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
            subject_indices.append(i)

    # Security to not have an error if no nouns or pronouns are found
    if len(subject_indices) == 0:
        # return a  random selection of a random number of word from a text
        return " ".join(random.sample(text.split(), random.randint(1, min(len(text.split()), 30)))) + " ?"
    
    to_keep = random.sample(subject_indices, random.randint(1, min(len(subject_indices), 5)))

    fake_question = " ".join([t for i, t in enumerate(tokens) if i in to_keep])
    
    return fake_question + ' ?'

def switch_subject(question: str) -> str:
    """
    Applies a switch augmentation by switching the positions
    of nouns or pronouns in the question.

    Args:
        text: A string containing the question to modify.

    Returns:
        A string containing the new question with the switched nouns.
    """
    # Tokenize the question
    tokens = nltk.word_tokenize(question)

    # Part of speech tag the tokens
    pos_tags = nltk.pos_tag(tokens)
    subject_indices = []
    for i, (token, pos) in enumerate(pos_tags):
        if pos in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
            subject_indices.append(i)
    
    if len(subject_indices) >= 2 :
        inds = random.sample(subject_indices, 2)
        to_switch = [tokens[u] for u in inds]
        for i, u in enumerate(inds[::-1]):
            tokens[u] = to_switch[i]
        
        return " ".join([t for i, t in enumerate(tokens)])
    
    else:
        return ' '.join(tokens[1:-1][::-1]) + ' ?'
    
    # Now 0 is for False and 1 for True
def corrupt_and_convert(batch: List[Dict[str, Any]], corruption_rate: float = 0.2) -> List[Dict[str, Any]]:
    """
    This function corrupts and converts a batch of data to improve the robustness of
    a natural language processing model. It applies various data augmentation 
    techniques to the input text to create a new dataset with modified examples.
    The function also adds unanswerable questions to the dataset for the model to 
    learn to identify such questions.

    Args:
        batch: A list of dictionaries where each dictionary contains a text, its language,
        a question, and if the question is answerable.

        corruption_rate: A float value between 0 and 1 that determines the rate of data
        augmentation applied to the input text.

    Returns:
        A list of dictionaries where each dictionary contains a text (including its
        language, a question and the context) and the target is a binary value representing
        whether the instance is answerable (1) or unanswerable (0).
    """
    if corruption_rate > 0.:
        eda = EasyDataAugmenter(pct_words_to_swap=0.2, transformations_per_example=1)
    new_data = []
    ready = False
    for i, data in enumerate(batch):
        l = data['language'] + ' '
        # l = 'english :' if data['language']=='<en>' else 'french :' # it was for a test without special tokens
        # Pass if we swithed two context so it's already in new_data
        if ready:
            ready = False
            pass
        
        # Convert and add the unanswerable question
        elif not data['answerable']:
            context = l+ data['question'] + ' </s> ' + data['text']
            new_data.append({'input': context, 'target': 0})            
        
        # Randomly apply one of the four corruption function
        else:
            if random.random() > 1 - corruption_rate:
                p = random.random()
                if p < 0.4 and i+1<len(batch):
                    for d in swap_2(data, batch[i+1]):
                        new_data.append(d)
                    ready = True                
                elif p < 0.6:
                    if data['language'] == '<en>':
                        # noise with testattack
                        context = l+ eda.augment(data['question'])[0] + ' </s> ' + data['text']
                        new_data.append({'input': context, 'target': 0})
                    elif data['language'] == '<fr>':
                        # We take the other noise functions for the moment
                        p = random.uniform(0.6, 1)
                elif p < 0.7:
                    context = l+ crop_words(data['question']) + ' </s> ' + data['text']
                    new_data.append({'input': context, 'target': 0})
                elif p < 0.8:
                    context = l+ remove_subjects(data['question']) + ' </s> ' + data['text']
                    new_data.append({'input': context, 'target': 0})
                elif p < 0.9:
                    context = l+ noun_question(data['text']) + ' </s> ' + data['text']
                    new_data.append({'input': context, 'target': 0})
                else:
                    context = l+ switch_subject(data['question']) + ' </s> ' + data['text']
                    new_data.append({'input': context, 'target': 0})

            # If no corruption, just add the data
            else:
                context = l+ data['question'] + ' </s> ' + data['text']
                new_data.append({'input': context, 'target': 1})
    return new_data



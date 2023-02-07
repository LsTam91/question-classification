import random
import nltk
# import stanza
# nlp = stanza.Pipeline(lang="fr")

def convert_data(data):
    new_data = []
    for i, a in enumerate(data):
        context = a['question'] + ' </s> ' + a['text']
        if a['answerable'] == True:
            target = 0
        else: target = 1
        
        new_data.append({'input': context, 'target': target})
    
    return new_data

def noise_2first(data):
    new_data = []
    
    context = data[1]['question'] + ' </s> ' + data[0]['text']
    new_data.append({'input': context, 'target': 1})
    
    context = data[0]['question'] + ' </s> ' + data[1]['text']
    new_data.append({'input': context, 'target': 1})

    for i in range(2, len(data)):
        context = data[i]['question'] + ' </s> ' + data[i]['text']
        if data[i]['answerable'] == True:
            target = 0
        else: target = 1
        
        new_data.append({'input': context, 'target': target})        

    return new_data

def noise_2(d1, d2):
    new_data = []
    
    context = d1['question'] + ' </s> ' + d2['text']
    new_data.append({'input': context, 'target': 1})
    
    context = d2['question'] + ' </s> ' + d1['text']
    new_data.append({'input': context, 'target': 1})

    return new_data

def crop_words(question):
    words = question.split()
    if len(words) < 4:
        return words[-1]
    num_words = random.randint(2, len(words) // 2 + 1)
    indexes = random.sample(range(len(words)), len(words) - num_words)
    indexes.sort()
    cropped_words = [words[i] for i in indexes]
    return ' '.join(cropped_words)

def remove_subjects(question):
    # Tokenize the question
    tokens = nltk.word_tokenize(question)

    # Part of speech tag the tokens
    pos_tags = nltk.pos_tag(tokens)

    # Find the indices of the two random nouns or pronouns in the question
    subject_indices = []
    for i, (token, pos) in enumerate(pos_tags):
        if pos in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
            subject_indices.append(i)
    
    # Choose between 1 to 3 of them to remove
    num_words = random.randint(1, len(subject_indices) // 2 + 1)
    
    # Pour éviter toute erreur
    if subject_indices == []:
        return ' ?'
    to_crop = random.sample(subject_indices, num_words)

    # Remove the subjects from the question
    question_without_subjects = " ".join([t for i, t in enumerate(tokens) if i not in to_crop])

    return question_without_subjects

def noun_question(question):
    # Tokenize the question
    tokens = nltk.word_tokenize(question)

    # Part of speech tag the tokens
    pos_tags = nltk.pos_tag(tokens)
    
    subject_indices = []
    for i, (token, pos) in enumerate(pos_tags):
        if pos in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
            subject_indices.append(i)

    if len(subject_indices) == 0:
        print(question)
        return ' ?' # ou question + ' ?'
    
    to_keep = random.sample(subject_indices, random.randint(1, min(len(subject_indices), 5)))

    fake_question = " ".join([t for i, t in enumerate(tokens) if i in to_keep])
    
    return fake_question + ' ?'

def switch_subject(question):
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
        
        # Changer la lettre capitale du debut de phrase si on prend le permier mot?
        return " ".join([t for i, t in enumerate(tokens)])
    
    else:
        return ' '.join(tokens[1:-1][::-1]) + ' ?'
    
    
def corrupt_and_convert(batch, language='fr', corruption_rate=0.2): # nlp, 
    new_data = []
    ready = False
    for i, data in enumerate(batch):
        l = data['language'] + ' '
        # Pass if we swithed two context so it's already in new_data
        if ready:
            ready = False
            pass
        
        # Convert and add the unanswerable question
        elif not data['answerable']:
            context = l+ data['question'] + ' </s> ' + data['text']
            new_data.append({'input': context, 'target': 1})            
        
        # Randomly apply one of the four corruption function
        else:
            if random.random() > 1 - corruption_rate:
                p = random.random()
                if p < 0.4 and i+1<len(batch):
                    for d in noise_2(data, batch[i+1]):
                        new_data.append(d)
                    ready = True
                elif p < 0.5:
                    context = l+ crop_words(data['question']) + ' </s> ' + data['text']
                    new_data.append({'input': context, 'target': 1})
                elif p < 0.65:
                    context = l+ remove_subjects(data['question']) + ' </s> ' + data['text']
                    new_data.append({'input': context, 'target': 1})
                elif p < 0.8:
                    context = l+ noun_question(data['text']) + ' </s> ' + data['text']
                    new_data.append({'input': context, 'target': 1})
                else:
                    context = l+ switch_subject(data['question']) + ' </s> ' + data['text']
                    new_data.append({'input': context, 'target': 1})

            # If no corruption, just add the data
            else:
                context = l+ data['question'] + ' </s> ' + data['text']
                new_data.append({'input': context, 'target': 0})
    return new_data


def corrupt(batch, corruption_rate=0.2): # nlp, 
    new_data = []
    ready = False
    for i, data in enumerate(batch):
        # Pass if we swithed two context so it's already in new_data
        if ready:
            ready = False
            pass
        
        # Convert and add the unanswerable question
        elif not data['answerable']:
            new_data.append({'question': data['question'], 'context': data['text'], 'target': 1})            
        
        # Randomly apply one of the four corruption function
        else:
            if random.random() > 1 - corruption_rate:
                p = random.random()
                if p < 0.33 and i+1<len(batch):
                    new_data.append({'question': data['question'], 'context': data['text'], 'target': 1})
                    new_data.append({'question': batch[i+1]['question'], 'context': batch[i+1]['text'], 'target': 1})
                    ready = True
                elif p < 0.66:
                    new_data.append({'question': crop_words(data['question']), 'context': data['text'], 'target': 1})
                else:
                    new_data.append({'question': remove_subjects(data['question']), 'context': data['text'], 'target': 1})
                # else:
                #     context = get_important_noun_phrases(data['text'], nlp=nlp) + ' </s> ' + data['text']
                #     new_data.append({'input': context, 'target': 1})

            # If no corruption, just add the data
            else:
                new_data.append({'question': data['question'], 'context': data['text'], 'target': 0})
    return new_data


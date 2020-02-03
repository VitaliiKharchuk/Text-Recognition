from collections import defaultdict
from pathlib import Path
from sklearn import metrics


path_sep = '/'  # '\\' for Windows, '/' for *nix

punctuations = ',;:-'
sentence_end = '.!?'
topwords = [['the'], ['be', 'is', 'am', 'are', 'was', 'were', 'been'], ['to'],
            ['of'], ['and'], ['a'], ['in'], ['that'], ['have', 'has', 'had'],
            ['i'], ['it'], ['for'], ['not'], ['or'], ['with'], ['he'], ['as'],
            ['you'], ['do', 'does', 'did', 'done'], ['at'], ['this'], ['but'],
            ['his'], ['by'], ['from'], ['they'], ['we'], ['say', 'says', 'said'],
            ['her'], ['she'], ['or'], ['an'], ['will'], ['my'], ['one', 'ones'],
            ['all'], ['would'], ['there'], ['their'], ['what'], ['so'], ['up'],
            ['out'], ['if'], ['about'], ['who'], ['get', 'gets', 'got', 'gotten'],
            ['which'], ['go', 'goes', 'went', 'gone'], ['me'], ['when'],
            ['make', 'makes', 'made'], ['can'], ['like', 'likes', 'liked'],
            ['time', 'times', 'timed'], ['no', 'noes'], ['just'], ['him'],
            ['know', 'knows', 'knew', 'known'], ['take', 'takes', 'took', 'taken'],
            ['people'], ['into'], ['year', 'years'], ['your'], ['good'], ['some'],
            ['could'], ['them'], ['see', 'sees', 'saw', 'seen'], ['other', 'others'],
            ['than'], ['then'], ['now'], ['look', 'looks', 'looked'], ['only'],
            ['come', 'comes', 'came'], ['its'], ['over'], ['think', 'thinks', 'thought'],
            ['also'], ['back'], ['after'], ['use', 'uses', 'used'], ['two'], ['how'],
            ['our'], ['work', 'works', 'worked'], ['first'], ['well'], ['way', 'ways'],
            ['even'], ['new'], ['want', 'wants', 'wanted'], ['because'], ['any'],
            ['these'], ['give', 'gives', 'gave', 'given'], ['day', 'days'], ['most'], ['us']]


def _finish_sentence(text, sentences, sentence, lastword, start):
    """Helper function to finish sentence creation (add last word and period etc.)
        Returns position of the beginning of the next sentence.
    """
    
    if len(lastword) > 0:
        sentence.append(lastword)
    if len(sentence) > 0:
        punct = text[start] if text[start] in sentence_end else '.'
        sentence.append(punct)
        sentences.append(sentence)
    start += 1
    while start < len(text):
        if text[start].isalpha():
            break
        start += 1
    return start if start == len(text) else (start - 1)


def _parse_text(text):
    """Read text and convert it to sequence of sentences."""
    
    sentences = []
    sentence, word = [], ''
    i = 0
    while i < len(text):
        c = text[i]
        if c.isalpha():
            word += c
        elif c.isspace():
            if len(word) > 0:
                sentence.append(word)
                word = ''
            if c == '\n' and text[i-1] == '\n':
                # Treat double new-line as an implicit end of sentence.
                i = _finish_sentence(text, sentences, sentence, word, i)
                sentence, word = [], ''
        elif c in sentence_end:
            i = _finish_sentence(text, sentences, sentence, word, i)
            sentence, word = [], ''
        elif c in punctuations:
            if len(word) > 0:
                sentence.append(word)
                word = ''
            sentence.append(c)
        else:
            # Skip any unknown symbols. Treat them as spaces.
            if len(word) > 0:
                sentence.append(word)
                word = ''
        i += 1
    # Finish last sentence.
    if len(sentence) > 0:
        sentence.append('.')
        sentences.append(sentence)
    return sentences


def _load_text(filename, merge=True, no_punct=True):
    # Open file and read text.
    try:
        f = open(filename, encoding='utf8')
    except FileNotFoundError:
        print('ERROR: Can not load text file:', filename)
        return None
    text = f.read()
    f.close()
    # Convert raw text into sequence of sentences.
    sentences = _parse_text(text.lower())
    # Remove punctuations if requested so.
    if no_punct:
        for i in range(len(sentences)):
            sentences[i] = [w for w in sentences[i] if w[0].isalpha()]
    # Merge tokens of a sentence into one long string if requested so.
    if merge:
        for i in range(len(sentences)):
            sentences[i] = ' '.join(sentences[i])
    return sentences


def read_books(path_to):
    """Read books.
        Two folder structures are supported:
        (A) Books of each author is in its own subfolder.
        (B) Books are in parent folder without author labeling.
    """
    
    books = {}
    for path in Path(path_to).glob('*'):
        if path.is_dir():
            path = str(path)
            author = path[path.rfind(path_sep)+1:]
            print('Author:', author)
            books[author] = {}
            for subpath in Path(path).glob('*'):
                if subpath.is_file() and str(subpath).endswith('.txt'):
                    subpath = str(subpath)
                    title = subpath[subpath.rfind(path_sep)+1 : subpath.rfind('.')]
                    print('        Book:', title)
                    books[author][title] = _load_text(subpath)
        elif path.is_file() and str(path).endswith('.txt'):
            path = str(path)
            title = path[path.rfind(path_sep)+1 : path.rfind('.')]
            print('Title:', title)
            books[title] = _load_text(path)
    return books


def train_model(classifier, feature_vector_train, label, feature_vector_valid=None, valid_y=None):
    """Train classifier on selected train dataset."""
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    if valid_y is not None:
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
        return metrics.accuracy_score(predictions, valid_y)
    return None


def classify(sentences, classifiers, transformers, method='intersect'):
    """Classifies text composed of selected sentences using selected
        classifiers and combining their results using selected method.
    """
    
    features = [t.transform(sentences) for t in transformers]
    predictions = [c.predict(f) for c, f in zip(classifiers, features)]
    labelcount = defaultdict(int)
    for i in range(len(sentences)):
        if len(predictions) == 1:
            labelcount[predictions[0][i]] += len(sentences[i])
        elif method == 'intersect':
            # Count only those labels that all predictors agrees on.
            ok = True
            for j in range(1, len(predictions)):
                if predictions[j][i] != predictions[0][i]:
                    ok = False
                    break
            if ok:
                labelcount[predictions[0][i]] += len(sentences[i])
        elif method == 'average':
            # Count labels from all predictors separately.
            for prediction in predictions:
                labelcount[prediction[i]] += len(sentences[i])
        elif method == 'most':
            # Count only those labels that most predictors agrees on.
            d = defaultdict(int)
            for prediction in predictions:
                d[prediction[i]] += len(sentences[i])
            d = sorted(d.items(), key=lambda x: -x[1])
            if len(d) == 1 or d[0][1] > d[1][1]:
                labelcount[d[0][0]] += d[0][1]
        else:
            print(f'ERROR: Unknonw combining method "{method}".')
            return None
    total = 0
    for _, count in labelcount.items():
        total += count
    probs = []
    for label in labelcount.keys():
        probs.append((labelcount[label] / total, label))
    return sorted(probs, reverse=True)
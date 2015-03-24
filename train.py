import sqlite3
from extractor import FeatureExtractor, Comment, Submission
from NeuralNetworks import MultilayerNetwork as MLN
from pprint import pprint
import math
import random
import matplotlib.pyplot as plt


def get_keywords(fname):
    with open(fname) as f:
        keywords = set(f.read().split())
    return keywords

def create_network():
    step = lambda x: x > 10
    step_prime = lambda x: 1
    sigmoid = lambda x: 1 / (1 + math.exp(-x))
    sigmoid_prime = lambda x: math.exp(x) / ((math.exp(x) + 1) ** 2)
    activation_functions = [[sigmoid for n in range(3)], [step]]
    function_derivatives = [[sigmoid_prime for n in range(3)], [step_prime]]

    weights = [
        [
            {'categories': 0.4477570002653856, 'keywords': 1.286670402570048, 'length': -1.33246684375284, 'link': 0.6045790303104172, 'title': 0.6341633775903553},
            {'categories': 1.6858316430238292, 'keywords': 1.754120978259333, 'length': 1.314913315069866, 'link': 1.2069882393200713, 'title': 1.1192226065835607},
            {'categories': 1.5283992851521815, 'keywords': 1.764098671646181, 'length': 1.264408223876475, 'link': 0.8266037645409774, 'title': 0.7115407340222178}
        ],
        [
            {0: 3.8460948693023647, 1: 4.5872259164475295, 2: 4.039869378507844}
        ]
    ]
    features = ['categories', 'keywords', 'length', 'link', 'title']
    net = MLN(features, activation_functions, function_derivatives, weights, alpha=0.05)
    return net

def get_data():
    query = 'SELECT id, comment_text, comment_author, submission_author, url, title, class FROM data_ext'
    cursor.execute(query)

    return cursor.fetchall()

def create_set(rows):
    testset = []
    for cid, body, cauthor, sauthor, url, title, doc_class in rows:
        submission = Submission(sauthor, url, title)
        comment = Comment(cid, cauthor, body, submission)
        features = extractor.extract(comment)
        testset.append( ([doc_class], features))
    return testset

def split_set(testset, ratio = 0.8):
    reviews  = [ (doc_class, features) for doc_class, features in testset if doc_class == [1]]
    comments = [ (doc_class, features) for doc_class, features in testset if doc_class == [0]]

    random.shuffle(reviews)
    random.shuffle(comments)

    reviewsplit = int(len(reviews) * ratio)
    commentsplit = int(len(comments) * ratio)

    testset = reviews[0 : reviewsplit] + comments[0 : commentsplit]
    trainingset = reviews[reviewsplit:] + comments[reviewsplit:]

    return trainingset, testset

if __name__ == '__main__':
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    
    keywords = get_keywords('keywords')
    extractor = FeatureExtractor(keywords=keywords, domains=['imgur'], mean_kw=36.165, mean_len=282.192)
    net = create_network()
    print(str(net))
    datasets = create_set(get_data())
    trainingset, testset = split_set(datasets)

    # keep track of the accuracies as the net is trained
    accuracies = []
    accuracies.append(float(net.test(testset)['accuracy']))

    decrease_counter = 0
    counter = 0
    max_accuracy = 0
    while decrease_counter < 128:
        random.shuffle(trainingset)
        for c, f in trainingset:
            net.train(c, f)
        test_results = net.test(testset)
        accuracy = float(test_results['accuracy'])
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            max_accuracy_iteration = counter
            max_accuracy = accuracy
            weights = net.weights()
            best_results = test_results
            decrease_counter = 0
        else:
            decrease_counter += 1
        counter += 1

    print('Found max after {} iterations.'.format(max_accuracy_iteration))
    print('Maximum accuracy after {} iterations:{}'.format(counter, max_accuracy))
    print('Using:')
    pprint(weights)
    print('# # #')
    plt.plot(range(len(accuracies)), accuracies)
    plt.show()
    # random.shuffle(testset)
    print('False Negatives:')
    pprint(best_results)

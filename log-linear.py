import os
import argparse
import numpy as np
import math
import json

from collections import defaultdict

class Feature:
    
    id = 0
    word_set = set()
    
    def __init__(self, path, label):
        self.gold = label
        self.spam_cntr, self.ham_cntr = self.from_file(path)
        self.id = Feature.id
        Feature.id += 1

    def real_cntr(self, label):
        return self.ham_cntr if label == 0 else self.spam_cntr
        
    def other_cntr(self, label):
        return self.spam_cntr if label == 0 else self.ham_cntr
            
    def from_file(self, path):
        spam_cntr, ham_cntr = defaultdict(int), defaultdict(int)
        with open(path, 'r', encoding='ISO-8859-1') as f:
            tokens = f.read().strip().lower().split()
        for e, word in enumerate(tokens):
            if e < len(tokens)-1:
                bigramm = f'{word} {tokens[e+1]}'
                bigramm_spam = f'{bigramm}-{str(1)}'
                bigramm_ham = f'{bigramm}-{str(0)}'
                Feature.word_set.add(bigramm_ham)
                Feature.word_set.add(bigramm_spam)
                spam_cntr[bigramm_spam] += 1
                ham_cntr[bigramm_ham] += 1
            word_ham = f'{word}-{str(0)}'
            word_spam = f'{word}-{str(1)}'
            ham_cntr[word_ham] += 1
            spam_cntr[word_spam] += 1
            Feature.word_set.add(word_ham)
            Feature.word_set.add(word_spam)
        return spam_cntr, ham_cntr
        
class Vectorizer:
    
    # Maps files in directory to feature fectors
    
    def __init__(self, path, data_begin, data_range):
        print('preprocessing dataset...')
        self.labels = {label:e for e, label in enumerate(['ham', 'spam'])}
        self.features = self.load_data(path, data_begin, data_range)
        np.random.shuffle(self.features)
    
    def file_path_gener(self, path, data_begin, data_range):
        path = path + 'enron' if path.endswith('/') else path + '/' + 'enron'
        subs = ['ham', 'spam']
        for i in range(data_begin, data_range):
            for label in subs:
                subpath = path + str(i) + '/' + label
                for _,_,files in os.walk(subpath):
                    for file in files:
                        yield (subpath + '/' + file, label)
        
    def load_data(self, path, data_begin, data_range):
        features = []
        for path, label in self.file_path_gener(path, data_begin, data_range):
            num_label = self.labels[label]
            feature = Feature(path, num_label)
            features.append(feature)
        return features

class LinClassifier:
    
    # a log-linear classifier

    def __init__(self, *args, **kwargs):
        if len(args) > 2 or len(kwargs) > 2:
            self.init_train(*args, **kwargs)
        else:
            self.init_test(*args, **kwargs)

    def init_train(self, mode, dir_path, learning_rate, epochs, regularization, load_path):
        self.train_vectorizer = Vectorizer(dir_path, 1, 5)
        self.dev_vectorizer = Vectorizer(dir_path, 5, 6)
        self.weights = {word:[0.0,0] for word in Feature.word_set}
        print("Initialize training...\n")
        if mode == 'optimize':
            learning_rate, regularization = self.optimize_hyperparams()
            self.reset()
        self.training(self.train_vectorizer.features, epochs, learning_rate, (1-learning_rate*regularization))
        print("Development Accuracy: ", self.accuracy(self.dev_vectorizer.features))
        self.to_json(load_path)
    
    def init_test(self, dir_path, load_path):
        self.weights = self.from_json(load_path)
        vectorizer = Vectorizer(dir_path, 6, 7)
        print("Test Accuracy: ", self.accuracy(vectorizer.features))
        
    def softmax(self, lst):
        result = []
        denom = self.logsumexp(lst)
        for c, score in lst:
            soft_score = math.exp(score - denom)
            result.append((c, soft_score))
        return result
    
    def logsumexp(self, lst):
        c = max(lst, key=lambda x: x[1])[1]
        return c + math.log(sum([math.exp(x - c) for _,x in lst]))
    
    def dot(self, cntr):
        return sum([val * self.weights.get(key, [0,0])[0] for key, val in cntr.items()])
            
    def get_probs(self, feature):
            scores = [(e,self.dot(cntr)) for e, cntr in enumerate([feature.ham_cntr, feature.spam_cntr])]
            probs = self.softmax(scores)
            return probs

    def classify(self, probs):
        return max(probs, key=lambda x: x[1])
    
    def training(self, train_set, epochs, lr, decay):
        counter = 0
        for i in range(epochs):
            for feature in train_set:
                counter += 1
                probs = self.get_probs(feature)
                self.sgd_update(feature, probs, lr, decay, counter)
            print("Accuracy at epoch {}: {}".format(i, self.accuracy(self.dev_vectorizer.features)))
        for key in self.weights:
            self.weights[key][0] = (decay**(counter - self.weights[key][1]))*self.weights[key][0]
                  
    def sgd_update(self, feature, probs, lr, decay, counter):
        gold_cntr = feature.real_cntr(feature.gold)
        other_cntr = feature.other_cntr(feature.gold)
        for key in gold_cntr:
            expected_prob = sum([prob * gold_cntr[key] if c == feature.gold else prob * other_cntr[key] for c, prob in probs])
            self.weights[key][0] = (decay**(counter - self.weights[key][1]))*self.weights[key][0] + lr*(gold_cntr[key] - expected_prob)
            self.weights[key][1] = counter
            
    def accuracy(self, set, ):
        with open('./classification.txt', 'w') as w:
            w.write("Results: \n\n")
            correct, total = 0, 0
            for e,feature in enumerate(set):
                probs = self.get_probs(feature)
                result, _ = self.classify(probs)
                w.write("Datapoint {} : {} ..... {} \n".format(e, result, result == feature.gold))
                if result == feature.gold:
                    correct += 1
                total += 1
        return correct / total
    
    def to_json(self, path):
        with open(path, 'w') as j:
            json.dump(self.weights, j)
            print("weights saved at: ", path)
    
    def from_json(self, path):
        with open(path, 'r') as j:
            weights = json.load(j)
        return weights
    
    def optimize_hyperparams(self):
        best_lr, best_reg, best_acc = 0.0, 0.0, 0.0
        for lr in [0.0001, 0.0003, 0.0005, 0.0007, 0.001]:
            for reg in [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]:
                self.reset()
                self.training(self.train_vectorizer.features, 2, lr, (1-lr*reg))
                accuracy = self.accuracy(self.dev_vectorizer.features)
                if accuracy > best_acc:
                    print("New best accuracy:{} new best lr:{} new best l2:{}".format(accuracy, lr, reg) )
                    best_acc = accuracy
                    best_lr = lr
                    best_reg = reg
        with open('./best_hyperparameters.txt', 'w') as f:
            f.write("Best Learning Rate: {}\n".format(best_lr))
            f.write("Best Regularization: {}\n".format(best_reg))
        return best_lr, best_reg
    
    def reset(self):
        self.weights = {word:[0.0,0] for word in Feature.word_set}
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, help='type in "train", "test", or "optimize"')
    parser.add_argument('dir_path', type=str, help='type in the path to the train, dev and test files')
    parser.add_argument('load_path', type=str, help='set this to the location of the weights')
    parser.add_argument('--lr', type=float, default=0.001, help='set the learning rate factor for the training')
    parser.add_argument('--l2', type=float, default=0.01, help='set the regularization factor to avoid overfitting')
    parser.add_argument('--epochs', type=int, default=4, help='set the number of iterations over the training set')
    
    args = parser.parse_args()
    
    if args.mode == 'optimize' or args.mode == 'train':
        classifier = LinClassifier(mode=args.mode,
                                    dir_path=args.dir_path,
                                    learning_rate=args.lr, 
                                    epochs=args.epochs,
                                    regularization=args.l2,
                                    load_path=args.load_path)
    else:
        classifier= LinClassifier(dir_path=args.dir_path, load_path=args.load_path)
    
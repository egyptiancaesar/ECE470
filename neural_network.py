from sklearn.neural_network import MLPClassifier
import sys, pickle
import numpy as np
from os import path

#nn_config[0] = alpha , #nn_config[1] = max_iter , #nn_confog[2] = solver used
#nn_config[3] = # of hidden layers iterated over.
nn_config = [0.001, 200, 'adam', list(range(4,64,4))]

def usage():
    print('Expected 3 arguments, got {}'.format(len(sys.argv)))
    print('Usage: python3 neural_network.py <data_file> <tags>')

def output_usage():
    print('please Ensure Images/ Folder already exists in current Directory')

def generate_NNs(data, solver, alpha, nn_sizes, max_iter):
    x_train, y_train = data["training_data"], data["training_target"]
    x_test, y_test = data["test_data"], data["test_target"]
    clfs = []
    for size in nn_sizes:
        clf = MLPClassifier(solver=solver,
        	             hidden_layer_sizes = size,
        	             alpha = alpha,
        	             learning_rate = 'constant',
        	             random_state = 0,
        	             max_iter = max_iter )
        clf.fit(x_train, y_train)
        clfs.append(clf)
    
    train_scores = [clf.score(x_train, y_train) for clf in clfs]
    test_scores = [clf.score(x_test, y_test) for clf in clfs]
    
    i = test_scores.index(max(test_scores))
    return clfs[i], test_scores[i], nn_sizes[i], train_scores[i]

def get_best_NN(data, alpha, nn_sizes, max_iter):
	best_sgd_NN, best_sgd_score, best_sgd_attrs, sgd_tr = generate_NNs(data, 'sgd', alpha, nn_sizes, max_iter)
	best_adam_NN, best_adam_score, best_adam_attrs, adam_tr = generate_NNs(data, 'adam', alpha, nn_sizes, max_iter)
	
	print('\033[1m' + '  Criterion    Training    Best Score' + '\033[0m')
	print('     SGD         {0:.2f}         {1:.2f}'.format(sgd_tr, best_sgd_score))
	print('     Adam        {0:.2f}         {1:.2f}'.format(adam_tr, best_adam_score))
	
	if best_adam_score > best_sgd_score:
		return best_adam_NN, best_adam_score, best_adam_attrs, adam_tr
	return best_sgd_NN, best_sgd_score, best_sgd_attrs, sgd_tr
def main():

    if len(sys.argv) < 2:
        usage()
        exit()

    output_file = ''
    if '-n' in sys.argv:
        if not path.exists('Images'):
            output_usage()
            exit()
    else:
        print('For Graph Generation, please use the following:')
        print('python3 neural_network.py <data_file> <tags>')
    
    input_file = sys.argv[1]
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
        
    alpha = nn_config[0]
    max_iter = nn_config[1]
    opt_method = nn_config[2]
    nn_sizes = nn_config[3]
    
    best_network, best_score, best_nn_size, best_solver = get_best_NN(data, alpha, nn_sizes, max_iter)
        
if __name__ == '__main__':
    main()

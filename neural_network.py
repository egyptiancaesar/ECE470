from sklearn.neural_network import MLPClassifier
import sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from os import path

#nn_config[0] = alpha , #nn_config[1] = max_iter , #nn_confog[2] = solver used
#nn_config[3] = # of hidden layers iterated over.

nn_alphas = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001] 
nn_solvers = ['sgd', 'adam'] #automated, graphed
nn_max_iters = [100, 200, 300, 400, 500] 
nn_sizes = list(range(4,12,4)) #automated, graphed

nn_config = [0.001, 200, 'adam', nn_sizes]

def usage():
    print('Expected 3 arguments, got {}'.format(len(sys.argv)))
    print('Usage: python3 neural_network.py <data_file> <tags>')

def output_usage():
    print('please Ensure Images/ Folder already exists in current Directory')

def NNs_analysis(solver, alphas, sizes, train_scores, test_scores):
    s_fig, s_ax = plt.subplots()
    s_ax.set_xlabel("Number of Hidden Layers")
    s_ax.set_ylabel("Accuracy")
    s_ax.plot(sizes, train_scores, marker = 'o', label = "train")
    s_ax.plot(sizes, test_scores, marker = 'o', label = "test")
    s_ax.legend()
    plot_name = 'Images/' + solver +'__no_of_layers_vs_tr'
    plt.savefig('' + plot_name + '.png')

def generate_NNs(data, nn_solvers, alpha, nn_sizes, max_iter):
    x_train, y_train = data["training_data"], data["training_target"]
    x_test, y_test = data["test_data"], data["test_target"]
    sgd_clfs = []
    adam_clfs = []
    for nn_solver in nn_solvers:
        for size in nn_sizes:
            clf = MLPClassifier(solver = nn_solver,
        	                 hidden_layer_sizes = size,
        	                 alpha = alpha,
        	                 learning_rate = 'constant',
        	                 random_state = 0,
            	                 max_iter = max_iter )
            clf.fit(x_train, y_train)
            if nn_solver == 'sgd':
            	sgd_clfs.append(clf)
            else:
            	adam_clfs.append(clf)

    sgd_train_scores = [clf.score(x_train, y_train) for clf in sgd_clfs]
    sgd_test_scores = [clf.score(x_test, y_test) for clf in sgd_clfs]
    adam_train_scores = [clf.score(x_train, y_train) for clf in adam_clfs]
    adam_test_scores = [clf.score(x_test, y_test) for clf in adam_clfs]

    NNs_analysis(nn_solvers[0], alpha, nn_sizes, sgd_train_scores, sgd_test_scores)
    NNs_analysis(nn_solvers[1], alpha, nn_sizes, adam_train_scores, adam_test_scores)

    sgd_i = sgd_test_scores.index(max(sgd_test_scores))
    adam_i = adam_test_scores.index(max(adam_test_scores))
    
    sgd = [ sgd_clfs[sgd_i], sgd_train_scores[sgd_i], nn_sizes[sgd_i], sgd_test_scores[sgd_i] ]
    adam = [ adam_clfs[adam_i], adam_train_scores[adam_i], nn_sizes[adam_i], adam_test_scores[adam_i] ]
    
    return sgd, adam

def get_best_NN(data, alpha, nn_sizes, max_iter):

    best_sgd, best_adam = generate_NNs(data, nn_solvers, alpha, nn_sizes, max_iter)

    print('\033[1m' + '  Criterion    Training    Best Score   Best Size'+ '\033[0m')
    print('     SGD         {0:.2f}         {1:.2f}         {2:.2f}'.format(best_sgd[1], best_sgd[3], best_sgd[2]))
    print('     Adam        {0:.2f}         {1:.2f}         {2:.2f}'.format(best_adam[1], best_adam[3] ,best_adam[2]))

    if best_adam[3]> best_sgd[3]:
        return best_adam[0], best_adam[3], best_adam[2], best_adam[1]
    return best_sgd[0], best_sgd[3], best_sgd[2], best_sgd[1]
    
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
    
    iter_1 = len(nn_alphas) * len(nn_sizes) * len(nn_solvers) * nn_max_iters[0]
    iter_2 = len(nn_alphas) * len(nn_sizes) * len(nn_solvers) * nn_max_iters[1]
    iter_3 = len(nn_alphas) * len(nn_sizes) * len(nn_solvers) * nn_max_iters[2]
    iter_4 = len(nn_alphas) * len(nn_sizes) * len(nn_solvers) * nn_max_iters[3]
    iter_5 = len(nn_alphas) * len(nn_sizes) * len(nn_solvers) * nn_max_iters[4]
    total_iter = iter_1 + iter_2 + iter_3 +iter_4 + iter_5
    print('Total Number of Iterations = {}'.format(total_iter))
    
    alpha = nn_config[0]
    max_iter = nn_config[1]

    best_network, best_score, best_nn_size, best_solver = get_best_NN(data, alpha, nn_sizes, max_iter)

if __name__ == '__main__':
    main()

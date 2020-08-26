import sys, csv, random, pickle
from os import path

def usage():
    print('Expected X arguments, got {}'.format(len(sys.argv)))
    print('Usage: python3 process_data.py <training_%> <input_data>')

def output_usage():
    print('Please Ensure Data/ Folder already exists in current Directory')

#removes irregularities in data values (type errors)
def clean_data_set(data):
    for x in data:
        for i in range(len(x)):
            if x[i] == 'NA':
                x[i] = 0
    return [[float(x) for x in row] for row in data]

#split data from their labels
def format_data(clean_data):
    data, target = [], []
    for row in clean_data:
        data.append(row[:-1])
        target.append(row[-1])
    return data, target

def main():
    if len(sys.argv) < 4:
        usage()
        exit()
    if not path.exists('Data'):
        output_usage()
        exit()
    #training set size
    tr_size_percent = float(sys.argv[1])
    if (tr_size_percent > 1):
        tr_size_percent = tr_size_percent/100

    input_file = sys.argv[2] #unprocessed data file
    output_file = sys.argv[3] #processed data file
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        data_block = list(reader)
    
    data_titles = data_block.pop(0)
    clean_data = clean_data_set(data_block)
    
    test_data = []
    for i in range(int((1 - tr_size_percent) * len(clean_data))):
        test_data.append(clean_data.pop(random.randint(0,len(clean_data) - 1)))

    training_data, training_target = format_data(clean_data)
    test_data, test_target = format_data(test_data)
    
    processed_data = {
        "training_data": training_data,
        "training_target": training_target,
        "test_data": test_data,
        "test_target": test_target,
        "feature_names": data_titles
    }

    with open('Data/' + output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    print()
    print("Suuccesfully cleaned and split data into {}% Training!!".format(tr_size_percent*100))
    print("          Binary Data stored in {}".format('Data/'+output_file))
    print()
   # print(data_titles)
   # print(data_block[0])
   # print(clean_data[0])

if __name__ == "__main__":
    main();

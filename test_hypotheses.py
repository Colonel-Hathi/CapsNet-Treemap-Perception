import os
import matplotlib.pyplot as plt

from model import ModelTreemap
from data_handler import get_testimages


def test_all(dataset, ckpt):
    testlist = os.listdir(dataset)
    model = ModelTreemap("Treemap", output_folder=None)
    # Load the model
    model.load(ckpt)
    csv = ['Set,Acc,Loss']
    for folder in testlist:
        path = os.path.join(dataset, folder)
        csvdata = test(path, model, folder)
        csv.append(csvdata)
    write_csv(csv, 'results')

def test(dataset, model, testset):
    """
        Train the model
        **input: **
            *dataset: (String) Dataset folder to used
            *ckpt: (String) [Optional] Path to the ckpt file to restore
    """

    # Get Test dataset
    X_test, y_test = get_testimages(dataset)
    X_test = X_test / 255

    # Evaluate all the dataset
    loss, acc, predicted_class = model.evaluate_dataset(X_test, y_test)

    print("Accuracy = ", acc)
    print("Loss = ", loss)

    stracc = str(acc)
    strloss = str(loss)
    csvdata = testset + ',' + stracc + ',' + strloss
    print('CSVDATA:' + csvdata)
    return csvdata


def write_csv(data, filename):
    print(data)
    with open('testresults/' + filename + '.csv', 'w') as f:
        for line in data:
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    test_all('dataset/hypotheses-test', 'output-nodedata1/checkpoints/ckpt1')
    #arguments = docopt(__doc__)
    #test_all(arguments["<dataset>"], arguments["<ckpt>"])
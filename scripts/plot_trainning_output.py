import matplotlib.pyplot as plt
import numpy as np

def plot_data(data, label):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)    
    ax.set_title(label)
    ax.set_xlabel('epochs')
    ax.plot(data)
    plt.show()


def extract_train_test_loss(log):
    classification_loss = []
    regression_loss = []
    running_loss = []
    accuracy = []
    for line in log.readlines():
        if line.startswith('Epoch'):
            name = 'Classification loss:'
            bidx = line.find(name)
            try:
                cls_loss = float(line[bidx+len(name): bidx+len(name)+7].strip())
                print('cls_loss:', cls_loss)
            except:
                print('error occurs')

            name = 'Regression loss:'
            bidx = line.find(name)
            try:
                reg_loss = float(line[bidx+len(name): bidx+len(name)+7].strip())
                print('reg_loss:', reg_loss)
            except:
                print('reg loss error occurs')

            name = 'Running loss:'
            bidx = line.find(name)
            try:
                run_loss = float(line[bidx+len(name): bidx+len(name)+7].strip())
                print('run_loss:', run_loss)
            except:
                print('run loss error occurs')

            classification_loss.append(cls_loss)
            regression_loss.append(reg_loss)
            running_loss.append(run_loss)

        if line.startswith('person:'):
            bidx = line.find('person:')
            acc = float(line[bidx+len('person:'):].strip())
            print('accuracy:', acc)
            accuracy.append(acc)

    return classification_loss, regression_loss, running_loss, accuracy

import sys

def main():    
    log_file = sys.argv[1]
    log = open(log_file,'r')
   
    classification_loss, regression_loss, running_loss, accuracy = extract_train_test_loss(log)
    plot_data(classification_loss, 'classification_loss')
    plot_data(regression_loss, 'regression_loss')
    plot_data(running_loss, 'running_loss')
    plot_data(accuracy, 'accuracy')
    log.close()

if __name__ == '__main__':
    main()


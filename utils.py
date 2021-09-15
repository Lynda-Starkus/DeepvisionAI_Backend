
import matplotlib.pyplot as plt
import argparse
import shutil
from os import listdir, mkdir
from os.path import isfile, isdir, join
from distutils.dir_util import copy_tree

parser = argparse.ArgumentParser()
parser.add_argument("-test", "--test",nargs='?' ,help="select the video to be tested range from 1 to 36", const = 36, type = int)
args = parser.parse_args()


def draw_ROC(false_positive_rate, true_positive_rate):

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - DecisionTree')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if args.test:
        
    ucsdped = 'UCSDped1'
    dir_test = [f for f in listdir('UCSD_Anomaly_Dataset/'+ucsdped+'/Test/') if isdir(join('UCSD_Anomaly_Dataset/'+ucsdped+'/Test/', f))]
    dir_test.sort()

    shutil.rmtree("UCSD_Anomaly_Dataset.v1p2", ignore_errors = True)
    dir_test_new = dir_test[0:((int(args.test)-1))]
    for i in range(int(args.test)*2):
        shutil.copytree('UCSD_Anomaly_Dataset/'+ucsdped+'/Test/'+dir_test[i], "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/"+dir_test[i])
            


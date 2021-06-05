# from cv2 import cv2
import cv2
import numpy as np
from os import system
import pickle
import gzip
from nn import cnn

model_path = "model/"
weight = 44
height = 44

with gzip.open("data/test.pkl.gz",'rb') as f:
    test = pickle.load(f)

def testNN(testset, model_path,show_num=False):

    with gzip.open(model_path,'rb') as f:
        net = pickle.load(f)

    graph_page = np.array([])
    graph_row = np.array([])
    num_page = np.array([])
    num_row = np.array([])
    rnum_page = np.array([])
    rnum_row = np.array([])

    cnt = 0
    hits = 0
    for x, y in testset:
        nn_output = np.argmax(net.feedforward(x))
        if (nn_output == y):
            hits += 1
            x_img = x
        else:
            x_img = 1 - x
        x_img = np.reshape(x_img,(28,28))
        if(cnt % weight == 0):
            if(cnt == weight and weight > 1):
                graph_page = graph_row
                num_page = num_row
                rnum_page = rnum_row
            elif(cnt == weight*height):
                if(height > 1):
                    graph_page = np.vstack((graph_page, graph_row))
                    num_page = np.vstack((num_page, num_row))
                    rnum_page = np.vstack((rnum_page, rnum_row))
                if(show_num):
                    system("cls")
                    same = num_page == rnum_page
                    num_page[same] = -1
                    rnum_page[same] = -1
                    print("NN output:")
                    print(num_page)
                    print("Correct:")
                    print(rnum_page)
                cv2.imshow("digit", graph_page)
                key = cv2.waitKey(0)
                if(key == 27): # Type ESC to break
                    break
                cnt = 0
            else:
                graph_page = np.vstack((graph_page, graph_row))
                num_page = np.vstack((num_page, num_row))
                rnum_page = np.vstack((rnum_page, rnum_row))
            graph_row = x_img
            num_row = nn_output
            rnum_row = y
        else:
            graph_row = np.hstack((graph_row, x_img))
            num_row = np.append(num_row, nn_output)
            rnum_row = np.append(rnum_row, y)
        cnt += 1
    cv2.destroyAllWindows()
    return hits,len(testset)

if (__name__ == "__main__"):
    with open(model_path+"index",'r') as f:
        model_type, model_name = f.read().splitlines()
    print(model_name)
    hits,all = testNN(test,model_path+model_type+'/'+model_name+"/model.pkl.gz")
    print("Accuracy on testset: %d / %d" % (hits, all))
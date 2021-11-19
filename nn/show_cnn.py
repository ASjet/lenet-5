import cv2
import numpy as np
from os import system
import pickle
import gzip
from nn import cnn, monite
from cv import ip

model_path = "model/"
testset_path = "data/test.pkl.gz"
weight = 44
height = 44


def loadData():
    with gzip.open(testset_path,'rb') as f:
        test = pickle.load(f)
    return test

def loadModel(model_name):
    with gzip.open(model_path+"cnn/"+model_name+"/model.pkl.gz",'rb') as f:
        net = pickle.load(f)
    return net

def testNN(model_name,show_num=False):

    with gzip.open(model_path+"cnn/"+model_name+"/model.pkl.gz",'rb') as f:
        net = pickle.load(f)
    net = loadModel(model_name)
    testset = loadData()

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
                    cv2.destroyAllWindows()
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


def showRes(model_name):
    testset = loadData()
    net = loadModel(model_name)

    for x, y in testset:
        src = np.reshape(x, (28,28))
        convd1 = net.cpl.conv(src, 1)
        poold1 = net.cpl.pooling(convd1, 1)
        convd2 = net.cpl.conv(poold1, 2)
        poold2 = net.cpl.pooling(convd2, 2)

        for i, l1 in enumerate(convd1):
            c1 = l1 if (i == 0) else np.vstack((c1, l1))

        for i, l2 in enumerate(poold1):
            p1 = l2 if (i == 0) else np.vstack((p1, l2))

        for i, l3 in enumerate(convd2):
            for j, ll3 in enumerate(l3):
                c2 = ll3 if (i == 0 and j == 0) else np.vstack((c2, ll3))

        for i, l4 in enumerate(poold2):
            for j, ll4 in enumerate(l4):
                p2 = ll4 if (i == 0 and j == 0) else np.vstack((p2, ll4))

        cv2.imshow("src", ip.zoom(src, 5))
        cv2.imshow("convd1", ip.zoom(c1, 5))
        cv2.imshow("poold1", ip.zoom(p1, 5))
        cv2.imshow("convd2", ip.zoom(c2, 5))
        cv2.imshow("poold2", ip.zoom(p2, 5))
        key = cv2.waitKey(0)
        if(key == 27):  # Type ESC to break
            break
    cv2.destroyAllWindows()


def showKernel(model_name):
    net = loadModel(model_name)
    for i, w in enumerate(net.cpl.weights1):
        w = ip.zoom(w, 5)
        if(i == 0):
            conv1 = w
        else:
            conv1 = np.hstack((conv1, w))
    for i,f in enumerate(net.cpl.weights2):
        for j,w in enumerate(f):
            w = ip.zoom(w, 5)
            if(j == 0):
                tmp = w
            else:
                tmp = np.hstack((tmp,w))
        if(i == 0):
            conv2 = tmp
        else:
            conv2 = np.vstack((conv2, tmp))

    cv2.imshow("weights1", conv1)
    cv2.imshow("weights2", conv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if (__name__ == "__main__"):
    with open(model_path+"index",'r') as f:
        model_type, model_name = f.read().splitlines()
    print(model_name)
    hits,all = testNN(test,model_path+model_type+'/'+model_name+"/model.pkl.gz")
    print("Accuracy on testset: %d / %d" % (hits, all))

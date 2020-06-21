"For practice of using OpenCV"

import cv2
import numpy as np
import math
import keras
import joblib
from z3 import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

def dataAugumentation(size):
    path = "SuDokuNumber/"
    x = []
    t = []
    erode = np.ones((2, 2), np.uint8)
    for i in range(0, 10):  # 数字
        for j in range(1, size+1):  # 番号
            # 初期設定
            img = cv2.imread(path+str(i)+"_"+str(j)+".jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
            img = cv2.GaussianBlur(img, (3, 3), 1)
            for k in range(-2, 3):  # 縦シフト
                for l in range(-4, 5):  # 横シフト
                    for m in [True, False]:  # erode
                        base = img.copy()
                        B1 = np.full((32, 32), 0, dtype='uint8')
                        if k < 0:
                            B1[:k] = base[-k:].copy()
                        elif k > 0:
                            B1[k:] = base[:-k].copy()
                        else:
                            B1 = base.copy()
                        dst = np.full((32, 32), 0, dtype='uint8')
                        if l < 0:
                            dst[:, :l] = B1[:, -l:].copy()
                        elif l > 0:
                            dst[:, l:] = B1[:, :-l].copy()
                        else:
                            dst = B1.copy()
                        if m:
                            dst = cv2.erode(dst, erode, iterations=1)
                        else:
                            pass
                        x.append(dst.copy())
                        t.append(i)
    joblib.dump(x, "x.joblib")
    joblib.dump(t, "t.joblib")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def trainModel(size, epc=5):
    x = np.array(joblib.load("x.joblib"))
    t = np.array(joblib.load("t.joblib"))

    p = np.random.permutation(len(x))
    x = x[p]
    t = t[p]

    # TrainData 90*10*size*0.9, TestData　90*10*size*0.1
    D = int(90*10*size*0.9)
    x_train = x[:D]
    y_train = t[:D]
    x_test = x[D:]
    y_test = t[D:]

    x_train1, x_valid, y_train1, y_valid = train_test_split(x_train, y_train, test_size=0.175)
    x_train = x_train1
    y_train = y_train1

    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], 32, 32, 1)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_valid /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_valid = keras.utils.to_categorical(y_valid, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=epc,
                        verbose=1,
                        validation_data=(x_valid, y_valid))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    joblib.dump(model, "CNN_for_MNIST.joblib")

def contrast(image, a):
    h, w = image.shape[0], image.shape[1]
    lut = [np.uint8(255.0 / (1 + math.exp(-a * (i - 128.) / 255.))) for i in range(256)]
    result_image = np.array([((lut[value]-1/2)*(g((i/w), i%w, h, w))+1/2) for i, value in enumerate(image.flat)], dtype=np.uint8)
    result_image = result_image.reshape(image.shape)
    return result_image

def f(x, y, h, w):  #0~1の値を返す。端程大きく、内側ほど小さくなる。exp(a((x-h/2)^2 + (y-w/2)^2 + b))
    k = 0
    a = (4-k)/(h**2+w**2)
    return a*((x-h/2)**2+(y-w/2)**2)+k

def g(x, y, h, w):  # exp(a((x-h/2)^2 + (y-w/2)^2 + b)), g(0,0) = 1, g(h/2,w/2) = d
    d = 1/6
    b = -(h**2+w**2)/4
    a = -4*math.log(d+1e-5)/(h**2+w**2)
    return math.exp(a*((x-h/2)**2 + (y-w/2)**2 + b))

def calculateArea(x):
    s = abs((x[1][0]-x[0][0])*(x[3][1]-x[0][1])-(x[3][0]-x[0][0])*(x[1][1]-x[0][1]))/2
    s += abs((x[1][0]-x[2][0])*(x[3][1]-x[2][1])-(x[3][0]-x[2][0])*(x[1][1]-x[2][1]))/2
    return s

def searchCorner(corner, h, w):
    """
    与えられたlistの中から最も頂点として適切な4点を返す
    """
    flag = False
    min_loss = 1000000
    ans = [[0]*2 for i in range(4)]
    for i in corner:
        xi, yi = i.ravel()
        if (0 < xi < h/2)and(0 < yi < w/2):
            for j in corner:
                xj, yj = j.ravel()
                if (h/2 < xj < h)and(0 < yj < w/2):
                    for k in corner:
                        xk, yk = k.ravel()
                        if (h/2 < xk < h)and(w/2 < yk < w): 
                            for l in corner:
                                xl, yl = l.ravel()
                                if (0 < xl < h/2)and(w/2 < yl < w):
                                    loss = 0

                                    len1 = ((xj-xi)**2+(yj-yi)**2)
                                    len2 = ((xl-xk)**2+(yl-yk)**2)
                                    loss += max(len1, len2) / min(len1, len2)  # 対辺の比率に対する罰則

                                    len3 = ((xj-xk)**2+(yj-yk)**2)
                                    len4 = ((xl-xi)**2+(yl-yi)**2)
                                    loss += max(len3, len4) / min(len3, len4)  # 対辺の比率に対する罰則

                                    loss += 3*max(max(len1, len2), max(len3, len4))/min(min(len1, len2), min(len3, len4))  # 最大長の辺と最小長の辺の比率に対する罰則

                                    theta1 = math.acos(-(yj-yi)/(((xj-xi)**2+(yj-yi)**2)**(1/2)))  # 左辺の垂直とのずれに対する罰則
                                    loss += abs(theta1-1.57)*40

                                    loss += 8*h*w/(calculateArea([[xi, yi], [xj, yj], [xk, yk], [xl, yl]])) -1
                                    if loss < min_loss:
                                        flag = True
                                        min_loss = loss
                                        ans = [[xi, yi], [xj, yj], [xk, yk], [xl, yl]]
    if flag:
        return ans

    return [0, 0]

def solveSuDoku(p):
    """
    数独のソルバー
    intのlistで渡す
    """
    solver = Solver()
    x = [[0]*9 for i in range(9)]
    for i in range(9):
        for j in range(9):
            x[i][j] = Int(str(9*i+j))
            if p[i][j] > 0:
                solver.add(x[i][j] == p[i][j])
            else:
                solver.add(x[i][j] > 0)
                solver.add(x[i][j] < 10)
    for i in range(9):
        solver.add(Distinct(x[i]))
        solver.add(Distinct([row[i] for row in x]))
        solver.add(Distinct(x[1][0],x[2][0]))
        solver.add(Distinct([x[i//3*3+j//3][i%3*3+j%3] for j in range(9)]))
    ans = [[0]*9 for i in range(9)]
    state = solver.check()
    if state == sat:
        res = solver.model() 
        for i in range(9):
            for j in range(9):
                ans[i][j] = res[x[i][j]].as_long()
    else:
        print(state)
    return ans

def captureAndSolve():  # 行列は↓x→y, 座標は↓y→x
    model = joblib.load("CNN_for_MNIST.joblib")
    cap = cv2.VideoCapture(1)
    
    if cap.isOpened():
        print("successfully opened")
        while True:
            _, frame = cap.read()
            #frame = cv2.flip(frame, 1)  # 左右反転
            h, w = frame.shape[0], frame.shape[1]
            cv2.rectangle(frame, (w//3, h//2-w//6), (w*2//3, h//2+w//6), (255, 0, 0))

            cv2.imshow("WINDOW", frame)

            key = cv2.waitKey(5)

            if key == ord('q'): # exit
                cv2.destroyWindow("WINDOW")
                break

            elif key == ord('x'): # decide
                x1, x2 = h//2-w//6+2, h//2+w//6-2
                y1, y2 = w//3+2, w*2//3-2

                # 切り出し、コーナー抽出
                board = frame[x1:x2, y1:y2, :]
                gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)  # グレースケール
                gray_copy = gray.copy()  # 数字認識用
                gray = contrast(gray, 17)  # コントラスト調整
                #cv2.imshow("contrast", gray)

                _, thres = cv2.threshold(gray, 200, 255, cv2.THRESH_TRUNC)  # 二値化
                #cv2.imshow("threshold", thres)

                corner = cv2.goodFeaturesToTrack(thres, 15, 0.01, 1)  # コーナー検出
                corner = np.int64(corner)
                sq = searchCorner(corner, thres.shape[0], thres.shape[1])  # 頂点を探索

                if len(sq) != 4:  # 頂点が定まらない
                    print("cannot recognize")

                else:
                    x_v = [sq[1][0]-sq[0][0], sq[1][1]-sq[0][1]]
                    y_v = [sq[2][0]-sq[1][0], sq[2][1]-sq[1][1]]
                    z_v = [sq[3][0]-sq[2][0], sq[3][1]-sq[2][1]]
                    w_v = [sq[0][0]-sq[3][0], sq[0][1]-sq[3][1]]

                    cv2.line(board, tuple(sq[0]), tuple(sq[1]), (0, 0, 255), 1)
                    cv2.line(board, tuple(sq[1]), tuple(sq[2]), (0, 0, 255), 1)
                    cv2.line(board, tuple(sq[2]), tuple(sq[3]), (0, 0, 255), 1)
                    cv2.line(board, tuple(sq[3]), tuple(sq[0]), (0, 0, 255), 1)

                    cv2.line(board, (sq[0][0]+x_v[0]//3, sq[0][1]+x_v[1]//3), (sq[2][0]+2*z_v[0]//3, sq[2][1]+2*z_v[1]//3), (0, 0, 255), 1)
                    cv2.line(board, (sq[0][0]+2*x_v[0]//3, sq[0][1]+2*x_v[1]//3), (sq[2][0]+z_v[0]//3, sq[2][1]+z_v[1]//3), (0, 0, 255), 1)
                    cv2.line(board, (sq[1][0]+y_v[0]//3, sq[1][1]+y_v[1]//3), (sq[3][0]+2*w_v[0]//3, sq[3][1]+2*w_v[1]//3), (0, 0, 255), 1)
                    cv2.line(board, (sq[1][0]+2*y_v[0]//3, sq[1][1]+2*y_v[1]//3), (sq[3][0]+w_v[0]//3, sq[3][1]+w_v[1]//3), (0, 0, 255), 1)

                    #cv2.imshow("board", board)

                    _, BW = cv2.threshold(gray_copy, 0, 255, cv2.THRESH_OTSU)
                    BW = cv2.bitwise_not(BW)
                    BW = cv2.GaussianBlur(BW, (3, 3), 1)
                    cv2.imshow("BW", BW)
                    problem = [[0]*9 for i in range(9)]
                    vec = lambda arg1, arg2: [-(1-arg2/9)*arg1/9*w_v[0]+arg2/9*(x_v[0]+arg1/9*y_v[0]), -(1-arg2/9)*arg1/9*w_v[1]+arg2/9*(x_v[1]+arg1/9*y_v[1])]  # i,jまでのベクトル
                    for i in range(9):
                        for j in range(9):
                            num = BW[int(sq[0][1]+vec(i, j)[1]) : int(sq[0][1]+vec(i+1, j)[1]), int(sq[0][0]+vec(i, j)[0]) : int(sq[0][0]+vec(i, j+1)[0])]
                            #num = BW[int(sq[0][1]-i*w_v[1]/9) : int(sq[0][1]-(i+1)*w_v[1]/9), int(sq[0][0]+j*x_v[0]/9) : int(sq[0][0]+(j+1)*x_v[0]/9)]
                            num = num[num.shape[0]//2-16:num.shape[0]//2+16, num.shape[0]//2-16:num.shape[0]//2+16]
                            #cv2.imshow("28num"+str(i)+str(j), num)
                            #cv2.imwrite("SuDokuNumber/a"+str(i)+str(j)+".jpg", num)
                            num = num.reshape(1, 32, 32, 1)
                            problem[i][j] = int(model.predict_classes(num)[0]%10)
                    #print(problem)
                    ans = solveSuDoku(problem)
                    for i in range(9):
                        for j in range(9):
                            if problem[i][j] == 0:
                                cv2.putText(board, str(ans[i][j]), (int(sq[0][0]+vec(i, j)[0])+8, int(sq[0][1]+vec(i+1, j)[1])-8), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
                    #print(ans)
                    cv2.imshow("ANSWER", board)
                #終了
                key = cv2.waitKey(0)
                while key != ord('q'):
                    key = cv2.waitKey(0)
                break
        cap.release()

    else:
        print("could not open")

#dataAugumentation(23)
#trainModel(23)
captureAndSolve()

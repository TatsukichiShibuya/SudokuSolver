import math
import numpy as np
import joblib
import cv2
import z3

class SudokuSolver:
    def solve(self, image):
        model = joblib.load("model/model.joblib")
        img4result = image.copy()
        img4edge = cv2.cvtColor(img4result, cv2.COLOR_BGR2GRAY)
        img4number = img4edge.copy()

        #頂点の抽出
        img4edge = self.contrast(img4edge, 17)
        _, img4edge = cv2.threshold(img4edge, 200, 255, cv2.THRESH_TRUNC)
        corner = cv2.goodFeaturesToTrack(img4edge, 15, 0.01, 1)
        corner = np.int64(corner)
        sq = self.serchCorner(corner, img4edge.shape[0], img4edge.shape[1])
        if len(sq) != 4:
            print("問題が読み取れません")
            return None, img4result
        else:
            x_v = [sq[1][0]-sq[0][0], sq[1][1]-sq[0][1]]
            y_v = [sq[2][0]-sq[1][0], sq[2][1]-sq[1][1]]
            z_v = [sq[3][0]-sq[2][0], sq[3][1]-sq[2][1]]
            w_v = [sq[0][0]-sq[3][0], sq[0][1]-sq[3][1]]

            cv2.line(img4result, tuple(sq[0]), tuple(sq[1]), (0, 0, 255), 1)
            cv2.line(img4result, tuple(sq[1]), tuple(sq[2]), (0, 0, 255), 1)
            cv2.line(img4result, tuple(sq[2]), tuple(sq[3]), (0, 0, 255), 1)
            cv2.line(img4result, tuple(sq[3]), tuple(sq[0]), (0, 0, 255), 1)

            cv2.line(img4result, (sq[0][0]+x_v[0]//3, sq[0][1]+x_v[1]//3), (sq[2][0]+2*z_v[0]//3, sq[2][1]+2*z_v[1]//3), (0, 0, 255), 1)
            cv2.line(img4result, (sq[0][0]+2*x_v[0]//3, sq[0][1]+2*x_v[1]//3), (sq[2][0]+z_v[0]//3, sq[2][1]+z_v[1]//3), (0, 0, 255), 1)
            cv2.line(img4result, (sq[1][0]+y_v[0]//3, sq[1][1]+y_v[1]//3), (sq[3][0]+2*w_v[0]//3, sq[3][1]+2*w_v[1]//3), (0, 0, 255), 1)
            cv2.line(img4result, (sq[1][0]+2*y_v[0]//3, sq[1][1]+2*y_v[1]//3), (sq[3][0]+w_v[0]//3, sq[3][1]+w_v[1]//3), (0, 0, 255), 1)
            cv2.imshow("grid", img4result)

        #文字認識
        _, img4number = cv2.threshold(img4number, 0, 255, cv2.THRESH_OTSU)
        img4number = cv2.bitwise_not(img4number)
        img4number = cv2.GaussianBlur(img4number, (3, 3), 1)
        problem = [[0]*9 for i in range(9)]
        vec = lambda arg1, arg2: [-(1-arg2/9)*arg1/9*w_v[0]+arg2/9*(x_v[0]+arg1/9*y_v[0]), -(1-arg2/9)*arg1/9*w_v[1]+arg2/9*(x_v[1]+arg1/9*y_v[1])]  # i,jまでのベクトル
        for i in range(9):
            for j in range(9):
                num = img4number[int(sq[0][1]+vec(i, j)[1]) : int(sq[0][1]+vec(i+1, j)[1]), int(sq[0][0]+vec(i, j)[0]) : int(sq[0][0]+vec(i, j+1)[0])]
                num = num[num.shape[0]//2-16:num.shape[0]//2+16, num.shape[0]//2-16:num.shape[0]//2+16]
                num = num.reshape(1, 32, 32, 1)
                problem[i][j] = int(model.predict_classes(num)[0]%10)

        #数独を解く
        ans = self.solveSAT(problem)
        if ans is not None:
            for i in range(9):
                for j in range(9):
                    if problem[i][j] == 0:
                        cv2.putText(img4result, str(ans[i][j]), (int(sq[0][0]+vec(i, j)[0])+8, int(sq[0][1]+vec(i+1, j)[1])-8), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            print("問題が読み取れません")
            return None, img4result

        return ans, img4result

    def solveSAT(self, problem):
        """
        args:
            problem : 9×9のintのarray、数独の問題
        return:
            ans : 9×9のintのarray、数独の解答
        """
        solver = z3.Solver()
        x = [[0]*9 for i in range(9)]
        for i in range(9):
            for j in range(9):
                x[i][j] = z3.Int(str(9*i+j))
                if problem[i][j] > 0:
                    solver.add(x[i][j] == problem[i][j])
                else:
                    solver.add(x[i][j] > 0)
                    solver.add(x[i][j] < 10)
        for i in range(9):
            solver.add(z3.Distinct(x[i]))
            solver.add(z3.Distinct([row[i] for row in x]))
            solver.add(z3.Distinct(x[1][0], x[2][0]))
            solver.add(z3.Distinct([x[i//3*3+j//3][i%3*3+j%3] for j in range(9)]))
        ans = [[0]*9 for i in range(9)]
        state = solver.check()
        if state == z3.sat:
            res = solver.model()
            for i in range(9):
                for j in range(9):
                    ans[i][j] = res[x[i][j]].as_long()
        else:
            print(state)
            ans = None
        return ans

    def serchCorner(self, corner, h, w):
        """
        args:
            corner : 頂点の候補
            h, w : 画像の縦横
        return:
            ans : 頂点(時計回り)、ない場合は[0, 0]
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

                                        loss += 8*h*w/(self.calculateArea([[xi, yi], [xj, yj], [xk, yk], [xl, yl]])) -1
                                        if loss < min_loss:
                                            flag = True
                                            min_loss = loss
                                            ans = [[xi, yi], [xj, yj], [xk, yk], [xl, yl]]
        if flag:
            return ans

        return [0, 0]

    def calculateArea(self, x):
        s = abs((x[1][0]-x[0][0])*(x[3][1]-x[0][1])-(x[3][0]-x[0][0])*(x[1][1]-x[0][1]))/2
        s += abs((x[1][0]-x[2][0])*(x[3][1]-x[2][1])-(x[3][0]-x[2][0])*(x[1][1]-x[2][1]))/2
        return s

    def contrast(self, image, a):
        h, w = image.shape[0], image.shape[1]
        lut = [np.uint8(255.0 / (1 + math.exp(-a * (i - 128.) / 255.))) for i in range(256)]
        result_image = np.array([((lut[value]-1/2)*(self.functoin2((i/w), i%w, h, w))+1/2) for i, value in enumerate(image.flat)], dtype=np.uint8)
        result_image = result_image.reshape(image.shape)
        return result_image

    def function1(self, x, y, h, w):  #0~1の値を返す。外側ほど大きく、内側ほど小さくなる。exp(a((x-h/2)^2 + (y-w/2)^2 + b))
        k = 0
        a = (4-k)/(h**2+w**2)
        return a*((x-h/2)**2+(y-w/2)**2)+k

    def functoin2(self, x, y, h, w):  # exp(a((x-h/2)^2 + (y-w/2)^2 + b)), g(0,0) = 1, g(h/2,w/2) = d
        d = 1/6
        b = -(h**2+w**2)/4
        a = -4*math.log(d+1e-5)/(h**2+w**2)
        return math.exp(a*((x-h/2)**2 + (y-w/2)**2 + b))

import glob
import numpy as np
import cv2
import joblib

def augument():  # 90倍に拡張
    x = []
    t = []
    erode = np.ones((2, 2), np.uint8)
    for i in range(0, 10):  # 数字
        imgs = glob.glob("image/"+str(i)+"_*")
        for path in imgs:
            img = cv2.imread(path)
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
    joblib.dump(x, "traindata/x.joblib")
    joblib.dump(t, "traindata/t.joblib")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    augument()

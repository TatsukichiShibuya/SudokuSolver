import glob
import cv2
import sudokuSolver as ss
import train as tr
import dataAugment as da

def solve():
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        print("opened")

        while True:
            _, frame = cap.read()
            h, w = frame.shape[0], frame.shape[1]
            x1, x2, y1, y2 = h//2-w//6, h//2+w//6, w//3, w*2//3
            cv2.rectangle(frame, (y1, x1), (y2, x2), (255, 0, 0))
            cv2.imshow("window", frame)

            key = cv2.waitKey(5)
            if key == ord('x'):
                board = frame[x1+2:x2-2, y1+2:y2-2, :]
                solver = ss.SudokuSolver()
                answer, result = solver.solve(board)

                if answer is not None:
                    cv2.imshow("answer", result)
                    answer = '\n'.join(map(str, [" ".join(map(str, i))  for i in answer]))
                    print(answer)

                    save = input("save result?(y/n) -> ")
                    if save == "y":
                        num = glob.glob("answer/*")
                        txt = open("answer/answer"+str(len(num)//2)+".txt", 'w')
                        txt.write(answer)
                        txt.close()
                        cv2.imwrite("answer/answer"+str(len(num)//2)+".jpg", result)

                key = cv2.waitKey(0)
                while key != ord('q'):
                    key = cv2.waitKey(0)
                break
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.release()
    else:
        print("failed")

def main():
    while True:
        mode = input("train or solve? -> ")
        if mode == "train":
            da.augument()
            tr.trainModel()
        elif mode == "solve":
            solve()
        else:
            print("invalid input")
            continue
        break

if __name__ == "__main__":
    main()

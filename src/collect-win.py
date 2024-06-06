import os
from sys import argv  #, stdin, stdout
# from tty import setcbreak

import cv2 as cv

if __name__ == '__main__':
    # this function isn't supported on windows
    # setcbreak(stdin.fileno())
    if len(argv) < 3:
        # save path example: "C:/collected"
        # modes are "m" for manual key presses and "v" for video capture.
        # limit is for setting the max number of images taken.
        print(f"Usage: {argv[0]} <save path> <mode> [limit]")
        exit(1)
    if len(argv) >= 4:
        image_limit = int(argv[3])
    else:
        image_limit = 0

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        exit(1)
    if not os.path.exists(argv[1]):
        os.mkdir(argv[1])

    i = 0
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not image_limit == 0 and i >= image_limit:
            print("\nStopped program on image_limit.")
            break

        # manual mode, press "e" to save image, "q" to quit.
        if argv[2] == "m":
            cv.imshow("0", frame)
            key = cv.waitKey(0) & 0xFF
            if key == ord("e"):
                cv.imwrite(os.path.join(argv[1], f"{i}.png"), frame)
                print(f"\rCreated {i}.png", end="")
                i += 1
            elif key == ord("q"):
                break
        # video mode, each frame of a video is collected and stored, press "q" to quit.
        elif argv[2] == "v":
            cv.imshow("0", frame)
            frame_count += 1
            if frame_count % 10 == 0:  # Change this number to capture at different intervals
                cv.imwrite(os.path.join(argv[1], f"{i}.png"), frame)
                print(f"\rCreated {i+1} images", end="")
                i += 1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    print()
    cap.release()
    cv.destroyAllWindows()

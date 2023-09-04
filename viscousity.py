import numpy as np
import cv2 as cv

def find_y(contour):
    flipedContour = np.flipud(contour) #Flip the contour matrix
    return np.argmax(flipedContour , axis= 0) # findout max value of fliped contour matrix
   

def find_du_dx(u):
    return np.diff(u)

    
if __name__ == '__main__':
    cap = cv.VideoCapture("Asserts/drop.mp4")
    fps = cap.get(cv.CAP_PROP_FPS)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    dt = 1/fps
    y0 = None
    y = None # y coordinate in pixels
    dy = None # dy coordinate in pixels
    u = None # u coordinate in pixels/sec
    du_dx = None
    du_dxDframe = None
    while True:
        
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        contour = cv.Canny(frame,100,255)
        if y is not None:
            y0 = y
        y = find_y(contour)
        if y0 is not None:
            dy = y - y0
            u = dy / dt
            du_dx = find_du_dx(u)
            if du_dxDframe is None:
                du_dxDframe = np.array([du_dx,])
            else:
                du_dxDframe = np.append(du_dxDframe,[du_dx],axis=0)


        cv.imshow('frame', contour)
        
        if cv.waitKey(1) == ord('q'):
            break
    print(du_dxDframe.shape)
    np.savetxt("du_dx.csv", du_dxDframe , delimiter=",")

    cap.release()
    cv.destroyAllWindows()

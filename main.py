from facial_orientation import FacialOrientation
import cv2

if(__name__ == '__main__'):
    cap = cv2.VideoCapture(0)
    if(cap is None or not cap.isOpened()):
            raise Exception("Warning: unable to open video source {}".format(cap))
    Face = FacialOrientation(cap=cap, angle_deviation=10, show_fps = False, show_coords = False, blur_background=True)
    try:
        is_save = False
        while(cap.isOpened() and not is_save):
            frame, is_save, message = Face.get_frame()
            print(message)
            cv2.imshow('Show', frame)
            if(cv2.waitKey(3) & 0xFF == 27):
                break
        cap.release()
    except Exception as e:
        print(e)
        
from facial_orientation import FacialOrientation

if(__name__ == '__main__'):
    Face = FacialOrientation()
    try:
        Face.run(cap = 0, angle_deviation = 9, show_fps = False, show_coords =  False, blur_background=True)
    except Exception as e:
        print(e)
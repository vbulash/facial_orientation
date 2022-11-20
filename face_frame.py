class FaceFrameCircle:
    def __init__(self, cx, cy, r):
        self.cx = int(cx)
        self.cy = int(cy)
        self.r = int(r)
    
    def point_in_face_frame(self, x, y):
        if(((self.cx - x)**2 + (self.cy - y)**2) < self.r**2):
            return True
        else:
            return False

    def points_in_face_frame(self, *points):
        for point in points:
            if(not self.point_in_face_frame(*point)):
                return False
        return True
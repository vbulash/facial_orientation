import cv2
import mediapipe as mp
import time
import numpy as np
from face_frame import FaceFrameCircle
import base64

RED = (77, 32, 238)
GREEN = (120, 214, 0)

class FacialOrientation:
    def __init__(self, cap, angle_deviation = 10, show_fps=False, show_coords=False, blur_background=False):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_incorrect_face_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=RED)
        self.drawing_correct_face_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=GREEN)
        self.counter_illumination = 0
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.is_save = False
        self.png = ''
        self.cap = cap
        self.show_fps = show_fps
        self.show_coords = show_coords
        self.blur_background = blur_background
        _, image = self.cap.read()
        img_h, img_w, img_c = image.shape
        self.face_frame_circle = FaceFrameCircle(img_w / 2, img_h / 2, min(img_w * 0.38, img_h * 0.38))

        self.angle_deviation = angle_deviation
        if(blur_background):
            self.mask = cv2.circle(np.zeros((img_h, img_w, img_c), dtype=np.uint8), (self.face_frame_circle.cx, self.face_frame_circle.cy), self.face_frame_circle.r, (255, 255, 255), -1)
        self.start_correct = 0
        self.illumination = 120
        self.is_correct_orientation = False 
        self.is_in_circle = False
        self.face_away = False

    def get_frame(self):
        success, image = self.cap.read()
        if(self.show_fps):
            start = time.time()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape
        image_original = image.copy()

        if(self.counter_illumination == 5):
            self.illumination = np.median(np.mean(image, axis = 2))
            self.counter_illumination = 0
        else:
            self.counter_illumination += 1
        if(self.illumination < 43):
            #cv2.putText(image, "Добавьте свет.", (int(img_h / 2), int(img_w / 2)), self.font, 1, RED, 1)
            return (image, self.is_save, "Добавьте свет.")
        elif(self.illumination > 200):
            #cv2.putText(image, "Уменьшите свет.", (int(img_h / 2), int(img_w / 2)), self.font, 1, RED, 1)
            return (image, self.is_save, "Уменьшите свет.")
        face_3d = []
        face_2d = []
        message = []
        
        if(results.multi_face_landmarks and not self.is_save):
            for face_landmarks in results.multi_face_landmarks:
                
                #Coordinates for face rectangle
                x_max = 0
                y_max = 0
                x_min = img_w
                y_min = img_h

                for idx, lm in enumerate(face_landmarks.landmark):
                    if(idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199):
                        if(idx == 1):
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                    x_temp, y_temp = int(lm.x * img_w), int(lm.y * img_h)
                    if(x_temp > x_max):
                        x_max = x_temp
                    if(x_temp < x_min):
                        x_min = x_temp
                    if(y_temp > y_max):
                        y_max = y_temp
                    if(y_temp < y_min):
                        y_min = y_temp
                
                x_max = x_max - 15
                x_min = x_min + 15
                y_max = y_max - 15
                y_min = y_min + 15

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                sucess, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                orientation = self.get_orientation(x, y)

                if(self.blur_background):
                    blurred_img = cv2.GaussianBlur(image, (21, 21), 0)
                    image = np.where(self.mask == np.array([255,255,255]), image, blurred_img)
                if(self.show_coords):
                    cv2.putText(image, "x: " + str(np.round(x,2)), (50, 50), self.font, 1, (0, 0, 0), 1)
                    cv2.putText(image, "y: " + str(np.round(y,2)), (50, 100), self.font, 1, (0, 0, 0), 1)
                    cv2.putText(image, "z: " + str(np.round(z,2)), (50, 150), self.font, 1, (0, 0, 0), 1)
                if(self.show_fps):
                    end = time.time()
                    totalTime = end - start
                    fps = 1 / totalTime
                    cv2.putText(image, f'FPS: {int(fps)}', (200, 450), self.font, 1.5, (0, 255, 0), 2)

                #Flags of conditions
                self.is_correct_orientation = orientation[0]
                self.is_in_circle = self.face_frame_circle.points_in_face_frame((x_min, y_min), (x_max, y_max), (x_min + (x_max - x_min), y_min), (x_min, y_min + (y_max - y_min)))
                self.face_away = (x_max - x_min < self.face_frame_circle.r*0.7 or y_max - y_min < self.face_frame_circle.r*0.7)

                if(not self.is_correct_orientation):
                    #textsize = cv2.getTextSize(orientation[1], self.font, 1, 1)
                    #textX = (img_w - textsize[0][0]) / 2
                    #textY = max(5, self.face_frame_circle.cy - self.face_frame_circle.r - textsize[1] - 10)
                    #cv2.putText(image, orientation[1], (int(textX), int(textY)), self.font, 1, RED, 1)
                    message.append(orientation[1])

                if(self.is_in_circle):
                    if(self.face_away):
                        #textsize = cv2.getTextSize("Лицо слишком далеко.", self.font, 1, 1)
                        #textX = (img_w - textsize[0][0]) / 2
                        #textY = min(img_h - textsize[1], self.face_frame_circle.cy + self.face_frame_circle.r + textsize[1] + 20)
                        #cv2.putText(image, "Лицо слишком далеко.", (int(textX), int(textY)), self.font, 1, RED, 1)
                        cv2.circle(image, (self.face_frame_circle.cx, self.face_frame_circle.cy), self.face_frame_circle.r, RED, 8)
                        message.append("Лицо слишком далеко.")
                    else:
                        cv2.circle(image, (self.face_frame_circle.cx, self.face_frame_circle.cy), self.face_frame_circle.r, GREEN, 2)
                else:
                    #textsize = cv2.getTextSize("Поместите лицо в окружность.", self.font, 1, 1)
                    #textX = (img_w - textsize[0][0]) / 2
                    #textY = min(img_h - textsize[1], self.face_frame_circle.cy + self.face_frame_circle.r + textsize[1] + 20)
                    cv2.circle(image, (self.face_frame_circle.cx, self.face_frame_circle.cy), self.face_frame_circle.r, RED, 8)
                    #cv2.putText(image, "Поместите лицо в окружность.", (int(textX), int(textY)), self.font, 1, RED, 1)
                    message.append("Поместите лицо в окружность.")

                if(orientation[0] and self.illumination > 40):
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.drawing_correct_face_spec,
                        connection_drawing_spec=self.drawing_correct_face_spec)
                else:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.drawing_incorrect_face_spec,
                        connection_drawing_spec=self.drawing_incorrect_face_spec)
        elif(not self.is_save):
            message.append("Нет лица.")
            #cv2.putText(image, "Нет лица.", (int(img_h / 2), int(img_w / 2)), self.font, 1, RED, 1)
        else:
            #textsize = cv2.getTextSize("Фотография сохранена!", self.font, 1, 1)
            #textX = (img_w - textsize[0][0]) / 2
            #textY = (img_h - textsize[0][1]) / 2
            #cv2.putText(image, "Фотография сохранена!", (int(textX), int(textY)), self.font, 1, GREEN, 1)
            message.append("Фотография сохранена.")
        if(self.is_correct_orientation and self.is_in_circle and not self.face_away and not self.is_save):
            self.start_correct +=1
            if(self.start_correct > 70 and not self.is_save):
                retval, buffer = cv2.imencode(".png", image_original[self.face_frame_circle.cy - self.face_frame_circle.r : self.face_frame_circle.cy + self.face_frame_circle.r,
                                                        self.face_frame_circle.cx - self.face_frame_circle.r : self.face_frame_circle.cx + self.face_frame_circle.r])
                self.png = base64.b64encode(buffer)
                self.is_save = True
                self.start_correct = 0
                return (self.png, self.is_save, message)
        else:
            self.start_correct = 0

        return (image, self.is_save, message)
                     
    def get_orientation(self, x, y) -> str:
        if(y < -self.angle_deviation):
            return (False, 'Поверните немного направо.')
        elif(y > self.angle_deviation):
            return (False, "Поверните немного налево.")
        elif(x < -self.angle_deviation):
            return (False, "Не опускайте голову.")
        elif(x > self.angle_deviation):
            return (False, "Не поднимайте голову.")
        else:
            return (True, "Корректное расположение.")
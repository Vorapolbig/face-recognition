import joblib
import cv2
import numpy as np
from PIL import Image
from face_recognition import preprocessing
from .util import draw_bb_on_img
from .constants import MODEL_PATH

import time



def main():
    cv2.namedWindow("video", cv2.WINDOW_NORMAL) 
    # cv2.resizeWindow('video',300,200)

    # cap = cv2.VideoCapture(-1)
    cap = cv2.VideoCapture('rtsp://192.168.10.47')
    # cap = cv2.VideoCapture('cam1.mp4')
    # cap = cv2.VideoCapture('test1.mov')
    
    face_recogniser = joblib.load(MODEL_PATH)
    preprocess = preprocessing.ExifOrientationNormalize()


    i = 0 
    sumFps = 0
    avgFps = 0
    playback = False
    unknownId = 0
    ratio = 8


    while True:

        timestr = time.strftime("%Y%m%d-%H%M%S")

        start = time.time()

        ret, frame = cap.read()

        new_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/ratio
        new_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)/ratio
        resolution = str(int(new_w)) + "X" + str(int(new_h)) + '/ FPS -> '+ str(int(avgFps))
        cv2.putText(frame,resolution,(50,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)
        if i == 0:
            print("resolution = "+ str(new_w) + "X" + str(new_h))

        resizedFrame = cv2.resize(frame, (int(new_w), int(new_h)))


        if i % 15 == 0:
            img = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
            resizeImg = Image.fromarray(img)
            faces = face_recogniser(preprocess(resizeImg))
            if faces is not None and faces:
                for face in faces:
                    if face.top_prediction.confidence > 0.5:
                        print("{} :  {} --> {}".format(timestr,face.top_prediction.label,
                        face.top_prediction.confidence))
                    else:
                        print(timestr + "Unknown person found: UnknownId =" + str(unknownId))
                        
                        #add margin 
                        paddedCrop = (face.bb.left * 0.7, face.bb.top * 0.7, face.bb.right * 1.3, face.bb.bottom * 1.3)
                        face_crop = resizeImg.crop(paddedCrop)
                        # face_crop = resizeImg.crop(face.bb)
                        face_crop.save('unknown_faces/'+timestr+'.jpg','jpeg')
                        unknownId += 1

        if playback and i > 0:
            imgOut = Image.fromarray(resizedFrame) 
            if faces is not None:
                draw_bb_on_img(faces, imgOut)

            # cv2.imshow('video', resizedFrame)
            cv2.imshow('video', np.array(imgOut))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # cv2.imshow('video', resizedFrame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        end = time.time()
        seconds = end-start
        fps = 1/seconds
        sumFps += fps
        if i % 50 == 0:
            avgFps = sumFps/50
            print("fps : {0}".format(avgFps))
            sumFps = 0
        i += 1
        


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
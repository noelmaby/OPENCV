import cv2
import mediapipe as mp
import time





cap=cv2.VideoCapture(0)


mpHands=mp.solutions.hands
hands = mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

pTime =0
cTime = 0



while True:
    ret,frame = cap.read()
    
    cTime=time.time()
    
    #hand drawing
    imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                print(id,lm)
                h,w,c = frame.shape
                cx , cy = int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id == 0 :
                    cv2.circle(frame,(cx,cy),25,(255,0,255),cv2.FILLED)
                    
                
                
            mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
    
    #fps
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime= cTime
    cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),
                2)
            
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.releae()
cv2.destroyAllWindows()


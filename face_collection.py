fileName=input("Enter Your Name:")
video=cv2.VideoCapture(0)
face_list=[]
classifier= cv2.CascadeClassifier("C:\\Users\\DELL\\AppData\\Local\\Programs\\Python\\Python38-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml") 
a=0
path="C:\\Users\\DELL\\Desktop\\python_file\\"
while True:
    a+=1
    check,frame=video.read()
    if check==False:
        continue
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=classifier.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5)
    if len(faces)==0:
        continue
    

    faces=sorted(faces,key=lambda x:x[2]*x[3])
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        select_face=frame[y-10:y+h+10,x-10:x+h+10]    #10 is offset
        select_face=cv2.resize(select_face,(100,100))
        face_list.append(select_face.reshape(-1))
        
    if len(face_list)==10:
        break
    
    cv2.imshow("Frame",frame)
    cv2.imshow("Face_Selected",select_face)
    
    key=cv2.waitKey(0) & 0xFF
    if key==ord("q"):
        break
#as np array
face_list=np.asarray(face_list)
face_list=face_list.reshape((face_list.shape[0],-1))
print(face_list.shape)

#save it as file
np.save(path+fileName+".npy",face_list)
print("sucessfully saved at this"+ path+fileName+".npy")
 
video.release()
cv2.destroyAllWindows()
    

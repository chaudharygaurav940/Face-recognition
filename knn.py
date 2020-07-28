dataset=pd.DataFrame(trainset)
dataset.rename(columns={30000:'labels'})
X,Y=dataset.iloc[:,1:-1],dataset.iloc[:,-1]
model=KNeighborsClassifier(n_neighbors=5,metric='euclidean')
model.fit(X,Y)
video=cv2.VideoCapture(0)
classifier=cv2.CascadeClassifier("C:\\Users\\DELL\\AppData\\Local\\Programs\\Python\\Python38-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml") 
a=0
while True:
    check,frame=video.read()
    
    if check==False:
         continue
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=classifier.detectMultiScale(gray,1.05,5)
    
    X_test=[]
    for face in faces[-1:]:
        x,y,w,h=face
        test_face=gray[y:y+h,x:x+w]
        test_face=cv2.resize(test_face,(100,100))
        X_test.append(test_face.reshape(-1))
    
        
        
        pred_value=model.predict(X_test)
        pred_name = names[int(pred_value)]
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("Image",frame)
    
    key = cv2.waitKey(0) & 0xFF
    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
        

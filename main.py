import cv2
import numpy as np
path=r'C:\Users\chinnu\OneDrive\Desktop\obj\6.jpg'
threshold=0.5

classes=["elephant"]
def detection(outputs,img):
    ht,wt,ct= img.shape
    bbox=[]
    classIds=[]
    confs=[]
    for output in outputs:
        for det in output:
            score = det[5:]
            classid = np.argmax(score)
            confidence = score[classid]
            if confidence > threshold:
                w,h = int(det[2]*wt),int(det[3]*ht)
                x,y = int((det[0]*wt)-w/2),int((det[1]*ht)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classid)
                confs.append(float(confidence))
    print(bbox)
    indices=list(cv2.dnn.NMSBoxes(bbox,confs,threshold,nms_threshold=0.2))
    print(indices)
    for i in indices:
        box=bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classes[classIds[i]]} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_PLAIN,0.9,(0,0,255),2)
        print(confs[i])

#cap= cv2.VideoCapture(0)
#sucess,img=cap.read()
img=cv2.imread(path)
#h,w,_ = img.shape
#print(h," ",w)
x=cv2.resize(img,(600,540))
y=cv2.resize(img,(600,540))
cv2.imshow('Input Image',x)
net=cv2.dnn.readNetFromDarknet('yolov4-test.cfg','yolov4-custom_best.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

blob=cv2.dnn.blobFromImage(img,1/255,(416,416),[0,0,0],1,crop=False)
net.setInput(blob)
layerNames=list(net.getLayerNames())
print(layerNames)

outputNames=[]
for i in layerNames:
    if i[0]=="y":
        outputNames.append(i)
print(outputNames)
outputs=net.forward(outputNames)
print(outputs[0].shape)
print(outputs[1].shape)
print(outputs[2].shape)
print(outputs[0][0])
detection(outputs,y)
cv2.imshow('Prediction',y)
cv2.waitKey(0)
cv2.destroyAllWindows()



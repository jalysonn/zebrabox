import numpy as np
import cv2
import argparse
import imutils


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 1)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

filename = "WIN_20180528_19_29_21_Pro.mp4"
classes_file = "obj.names"
weights = "yolo-obj_34200.weights"
config_file = "yolo-obj.cfg"
cap = cv2.VideoCapture(filename)

cont = 1

while(cap.isOpened()):
	ret, frame = cap.read()
		
	verde = (0, 255, 0)
####### QUADRANTES #######
	#canvas((x2,anguloHorario) , (x1,anguloAntiHorario),cor, espessuraBorda)
# 	cv2.line(frame, (1900,200), (0, 200), verde, 4)
# 	cv2.line(frame, (1900, 540), (0, 540), verde, 4)
# 	cv2.line(frame, (1900, 900), (0, 900), verde, 4)

# #LINHA VERTICAL
# #	canvas((anguloHorario,y1) , (anguloAntiHorario,y2),cor, espessuraBorda)
# 	cv2.line(frame, (700, 50), (700,1900), verde, 4)
# 	cv2.line(frame, (1060, 50), (1060,1900), verde, 4)
# 	cv2.line(frame, (1400, 50), (1450,1900), verde, 4)
	

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.resize(frame,None,fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC)
	edges = cv2.Canny(frame,100,200)

#### FIM DOS QUADRANTES

	crossingLine=np.zeros((2,2),np.float32)
	horizontalLinePosition=270
	crossingLine[0][0]= 270
	crossingLine[0][1]= horizontalLinePosition
	crossingLine[1][0]= 420
	crossingLine[1][1] =horizontalLinePosition

	cv2.line(frame,(crossingLine[0][0],crossingLine[0][1]),(crossingLine[1][0],crossingLine[1][1]),(0,255,0), 2)


	if cont % 100 == 0:
		Width = frame.shape[1]
		Height = frame.shape[0]
		scale = 0.00392

# read class names from text file
		classes = None
		with open(classes_file, 'r') as f:
			classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
		COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
		net = cv2.dnn.readNet(weights, config_file)

# create input blob
		blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)

# set input blob for the network
		net.setInput(blob)

# run inference through the network
# and gather predictions from output layers
		outs = net.forward(get_output_layers(net))

# initialization
		class_ids = []
		confidences = []
		boxes = []
		conf_threshold = 0.5
		nms_threshold = 0.4

# for each detetion from each output layer
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
						center_x = int(detection[0] * Width)
						center_y = int(detection[1] * Height)
						w = int(detection[2] * Width)
						h = int(detection[3] * Height)
						x = center_x - w / 10
						y = center_y - h / 10
						class_ids.append(class_id)
						confidences.append(float(confidence))
						boxes.append([x, y, w, h])
						

# apply non-max suppression
		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# go through the detections remaining
# after nms and draw bounding box
		for i in indices:
			
			i = i[0]
			box = boxes[i]
			x = box[0]
			y = box[1]
			w = box[2]
			h = box[3]

			quad = draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
			

#achar o centroide
			gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			ret,thresh = cv2.threshold(gray_image,127,255,0)

			cnt = frame[0]
			M = cv2.moments(cnt)
			try:
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
			except ZeroDivisionError:
				cX=0
				cY=0

			
			cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
			cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
			cv2.putText(frame, "center", (cX - 20, cY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


			# cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
			# cv2.putText(frame, "center", (cX - 20, cY - 20),
			# cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

			#v2.imshow('frame',frame)
			cv2.imshow('frame',frame)
#			cv2.imshow('frame', round(x))

			# font = cv2.FONT_HERSHEY_SIMPLEX
			# text =str(i)
			# cv2.putText(frame,"ob{}".format(text),(class_ids[i].centerPositions[-1][-2],blobs[i].centerPositions[-1][-1]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255 ,0), 2) 
			# cv2.putText(round(x),'opencv',(10,500), font , 1,(0,0,255),2,cv2.LINE_AA)
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	cont = cont + 1

cap.release()
cv2.destroyAllWindows()
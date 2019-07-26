from imutils.object_detection import non_max_suppression
import numpy as np
import cv2, time , os
from django.conf import settings

class Detector:
    p_image = []

    def set_image(self, img):
        """
         Fucntion for reading the image
        """
        self.image = cv2.imread(img, 1)

    def to_detect(self):
        self.t_detector()
        return self.o_detector()

    def t_detector(self):
        # Resize the image to 32 multiple size, and grab dimensions
        image = self.image
        image = cv2.resize(image, (960, 960))
        (H, W) = image.shape[:2]

        # Copy image to another variable
        orig = image.copy()


        ##############################################################
        ##                                                          ##
        ##             TEXT DETECTION CODES START HERE              ##
        ##                                                          ##
        ##                                                          ##
        ##############################################################

        # Load trained EAST model
        eastpath = settings.EAST_URL
        east = cv2.dnn.readNet(eastpath)

        #blob image and set it as an input, helps in the prediction of of area
        east.setInput(cv2.dnn.blobFromImage(image, 1.0, size=(H, W), swapRB=True, crop=False))

        # Start tracking the time used to detect text
        start = time.time()

        # Define two output layers, One for probability of text in the region
        # the other for deriving the bounding box
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ]

        #Forwad pass the image to otbtain the two layers
        (scores, geometry) = east.forward(layerNames)
        end = time.time()

    
        # Read rows and columns
        (rows, cols) = scores.shape[2:4]

        # Two empty lists, one for deteced regions(assuming rectangles)
        # other for probability

        rects = []
        probs = []
        # Loop over the rows
        for y in range(0, rows):
            # Extracts the scores
            scoresData = scores[0, 0, y]

            # Extracts the geometry data
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]

            # Extract the angle data
            anglesData = geometry[0, 4, y]

            # Loop over the columns
            for x in range(0, cols):

                # Check for sufficent probability of text in the region
                if scoresData[x] < 0.4:
                    continue

                # Compute the offset factor since the result features is 4x smaller than the image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # Extract the rotation angle of the detected region
                angle = anglesData[x]

                # Compute sine and cosine to get the coordinates
                sin = np.sin(angle)
                cos = np.cos(angle)

                # Derive the height and width of the bounding boxes
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # The starting and ending (x, y)-coordinates of the
                # text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                startX = int(endX - w)

                endY = int(offsetY + (sin * xData1[x]) + (cos * xData2[x]))
                startY = int(endY - h)

                # Append the bounding box coordinates and probability scores
                # to their respected lists
                rects.append((startX, startY, endX, endY))
                probs.append(scoresData[x])

        # Apply none-maxima to suppress weak detections
        boxes = non_max_suppression(np.array(rects), probs=probs)


        # Loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            #Draw bounding boxes
            cv2.rectangle(orig, (startX, startY), (endX, endY), (38, 160, 68), 3)
            cv2.putText(orig, "text", (startX , startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,255,255), 2)


        # Display the output
        orig = cv2.resize(orig, (int(orig.shape[1] / 1.5), int(orig.shape[0] / 1.5)))
        self.p_image = orig

    def o_detector(self):


        #############################################################
        #                                                          ##
        #             OBJECTS DETECTION CODES START HERE           ##
        #                                                          ##
        #                                                          ##
        #############################################################

        # Load COCO class label that yolo was trained on
        labelsPath = settings.YOLO_URL
        LABELS = open(labelsPath).read().strip().split("\n")

        # Assing a random color to each label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

        # Path to the YOLO WEIGHTS and YOLO CONF files

        weightsPath = settings.YOLO_WEIGHTS
        confPath = settings.YOLO_CONFIG

        # Load YOLO object detector trained on COCO dataset
        net = cv2.dnn.readNetFromDarknet(confPath, weightsPath)


        # Load the image (processed by the text_detector) and grab it's spatial dimensions
        image = self.p_image
        (H, W) = image.shape[:2]
        # if (H, W) > (1440, 900):
        #     image = cv2.resize(image, (620, 920))
        #     (H, W) = image.shape[:2]

        # Determine the only "output" layers needed from yolo
        layerName = net.getLayerNames()
        layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]


        # construct a blob from image
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, size=(416, 416), swapRB=True, crop=False)

        # Input blobed image to the detection process and forward pass through Yolo network
        net.setInput(blob)
        start = time.time()
        layersOutput = net.forward(layerName)
        end = time.time()

        # Yolo time
        self.yolo_time = end - start


        # VISUALIZE THE RESULTS

        # Initilize the list of detected bounding boxes, confidences and ClassIDs
        boxes = []          #Bounding boxes around the detected objects
        confidences = []    # The confidence value that yolo assings to an object
        objectIDs = []       # Detected objects class labels

        # Populate the above lists from the yolo outputs
        # Loop over each layer output

        for output in layersOutput:
            # Loop over each detection
            for detection in output:
                # Extract the classIDs and confidence
                scores = detection[5:]
                objectID = np.argmax(scores)
                confidence = scores[objectID]

                # Filter out the weak prediction

                if confidence > 0.5:

                    # Scale the bounding box back relative to the size of an image
                    # Remember: Yolo returns the center (x, y) coordinates followed
                    # by the boxes' width, height

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Using center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height  / 2))


                    # Update boxes, confidences and objectIDs lists
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    objectIDs.append(objectID)

        # Applying none maxima suppression to supress weak, overlapping bounding boxes
        pureBoxes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5) # 0.5 is probability value, and 0.3 is Threshold value

        # Ensure atleast one detection exists
        if len(pureBoxes) > 0:
            # Looping over the indexes
            for i in pureBoxes.flatten():

                # Extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # Draw the bounding box rectangle and label the image
                color = [int(c) for c in COLORS[objectIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                text = "{}: {:.2f}%".format(LABELS[objectIDs[i]].capitalize(),confidences[i] * 100)

                # cv2.rectangle(image, (x, y - 2), (x + 130, y - 20), color, -1)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,255,255), 2)

        # Return processed image
        return image

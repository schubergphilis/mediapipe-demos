from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui


class handTracker():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles

    def handsProcesser(self, image):

        # FMB: code below from the internet, still need to figure out if/how/why this
        # is needed.
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.image = image

        pass

    def handsDrawLandmarks(self, draw=True):
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        self.image, handLms, self.mpHands.HAND_CONNECTIONS,
                        self.mpDrawingStyles.get_default_hand_landmarks_style(),
                        self.mpDrawingStyles.get_default_hand_connections_style(),)
        pass

    def positionFinder(self, high_lighted_part=8):
      # FMB: obselete?
      # high_lighted_part = 8 means index finger
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                for id, lm in enumerate(hand.landmark):
                    if id == high_lighted_part:
                        h, w, c = self.image.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        cv2.circle(self.image, (cx, cy), 15,
                                   (255, 0, 255), cv2.FILLED,)

        pass

    def handsCountFingers(self):
        # FMB: Counts fingers, to be expanded with other gestures?

        # Initially set finger count to 0 for each cap
        fingerCount = 0

        if self.results.multi_hand_landmarks:

            for hand_landmarks in self.results.multi_hand_landmarks:
                # Get hand index to check label (left or right)
                handIndex = self.results.multi_hand_landmarks.index(
                    hand_landmarks)
                handLabel = self.results.multi_handedness[handIndex].classification[0].label

                # Set variable to keep landmarks positions (x and y)
                handLandmarks = []

                # Fill list with x and y positions of each landmark
                for landmarks in hand_landmarks.landmark:
                    handLandmarks.append([landmarks.x, landmarks.y])

                # Test conditions for each finger: Count is increased if finger is
                #   considered raised.
                # Thumb: TIP x position must be greater or lower than IP x position,
                #   deppeding on hand label.
                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    fingerCount = fingerCount+1
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    fingerCount = fingerCount+1

                # Other fingers: TIP y position must be lower than PIP y position,
                #   as image origin is in the upper left corner.
                if handLandmarks[8][1] < handLandmarks[6][1]:  # Index finger
                    fingerCount = fingerCount+1
                if handLandmarks[12][1] < handLandmarks[10][1]:  # Middle finger
                    fingerCount = fingerCount+1
                if handLandmarks[16][1] < handLandmarks[14][1]:  # Ring finger
                    fingerCount = fingerCount+1
                if handLandmarks[20][1] < handLandmarks[18][1]:  # Pinky
                    fingerCount = fingerCount+1

        self.fingerCount = fingerCount
        pass


def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    # Some handy stuff: a length 10 FIFO queue for detecting transitions between 1/0 and 2/0
    # We also make use of time.time() to generate a time out to prevent over triggering
    length_of_queue = 5 #frames
    time_out_period = 1 #seconds
    
    d = deque(np.zeros(length_of_queue))
    time_since_last_action = time.time()
    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        tracker.handsProcesser(image)
        tracker.handsDrawLandmarks()
        tracker.handsCountFingers()
        
        # Update queue
        d.appendleft(tracker.fingerCount)
        d.pop()
        # print(d)

        # Check if we captured hand gesture transition for next/previous slide
        if (time.time() - time_since_last_action) > time_out_period:
            if (np.min(list(d)) == 0) and (np.max(list(d)) == 1):
                # print("next slide")
                pyautogui.press("right")
                time_since_last_action = time.time()
            if (np.min(list(d)) == 0) and (np.max(list(d)) == 2):
                # print("previous slide")
                pyautogui.press("left")
                time_since_last_action = time.time()

        # Display the final results:
        # FMB: using tracker.image is apparantly important here, we pass on updated information
        # via the tracker.image object?
        cv2.putText(tracker.image, str(tracker.fingerCount), (50, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

        cv2.imshow('MediaPipe Hands', tracker.image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

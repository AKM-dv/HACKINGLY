import cv2
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np
import pygame
from time import sleep
import time
import threading
import mysql.connector
import matplotlib.pyplot as plt



LANDMARKS = []
PISTOL = []

db_connection = mysql.connector.connect(
            host="",
            user="",
            password="",
            database=""
        )
cursor = db_connection.cursor()




def run_for_seconds(seconds):
    t = threading.Thread(target=pistoldata)
    t.start()
    t.join(seconds)  # Wait for the thread to finish or until specified seconds elapsed
    if t.is_alive():
        print("Function running time exceeded {} seconds. Stopping.".format(seconds))
        t.join() 


def play_sound(sound_file):
    pygame.mixer.init()
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()



def pistoldata():
    global LANDMARKS,PISTOL
    start_time = time.time()  # Record the start time
    while time.time() - start_time < 5:  # Continue for 5 seconds
        if LANDMARKS:  # Check if there is data available
            PISTOL.append(LANDMARKS)  # Append data to the PISTOL list
        sleep(0.1)
    print(PISTOL)
    plotgraph(PISTOL)
    for data in PISTOL:
        angle, angle1, angle2, angle3 = data
        cmd = "INSERT INTO lm (angle, angle1, angle2, angle3) VALUES ({}, {}, {}, {})".format(angle, angle1, angle2, angle3)
        cursor.execute(cmd)
    db_connection.commit()

    


def mainchain():
    # we have to adjust for timings
    play_sound('TS.mp3')
    sleep(5)
    print("STAND STRAIGHT")
    play_sound('ss.mp3')
    sleep(9)
    print("TAKE GRANADE THROWING POSITION")
    play_sound('NP.mp3')
    sleep(8)
    run_for_seconds(6)
    


def plotgraph(BL):
    time_increment = 0.1

    # Calculate time values
    time_values = [i * time_increment for i in range(len(BL))]

    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(len(BL[0])):
        plt.plot(time_values, [values[i] for values in BL], label=f'Value {i+1}')

    # Plot ideal line
    ideal_line = [BL[0][0] + 5] * len(BL)
    plt.plot(time_values, ideal_line, '--', label='Ideal Upper Bound', color='Blue')
    plt.plot(time_values, [BL[0][0] - 5] * len(BL), '--', label='Ideal Lower Bound', color='Blue')

    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.title('Values over Time with Ideal Bounds')
    plt.legend()
    plt.grid(True)
    plt.show()




main = threading.Thread(target=mainchain)
main.start()

 # Wait until the thread finishes if it's still running

# Example usage: Run the function for 2 seconds



def calculate_angle(a, b, c):
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def open_camera():
    # global LANDMARKS
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    # Open the default camera (usually the first one)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect poses in the frame
        results = pose.process(frame_rgb)

        # Draw the landmarks on the frame
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            A = 16
            B = 12
            C = 13

            D = 14
            E = 12
            F = 24

            G = 11
            H = 13
            I = 15

            J = 16
            K = 14
            L = 12



           
            # Extracting landmarks for shoulder, elbow, and wrist
            A = [int(landmarks[A].x * frame.shape[1]), int(landmarks[A].y * frame.shape[0])]
            B = [int(landmarks[B].x * frame.shape[1]), int(landmarks[B].y * frame.shape[0])]
            C = [int(landmarks[C].x * frame.shape[1]), int(landmarks[C].y * frame.shape[0])]

            # A = [landmarks[A].x, landmarks[A].y]
            # B = [landmarks[B].x, landmarks[B].y]
            # C = [landmarks[C].x, landmarks[C].y]

            D = [landmarks[D].x, landmarks[D].y]
            E = [landmarks[E].x, landmarks[E].y]
            F = [landmarks[F].x, landmarks[F].y]

            
            G = [landmarks[G].x, landmarks[G].y]
            H = [landmarks[H].x, landmarks[H].y]
            I = [landmarks[I].x, landmarks[I].y]

            J = [landmarks[J].x, landmarks[J].y]
            K = [landmarks[K].x, landmarks[K].y]
            L = [landmarks[L].x, landmarks[L].y]

            cv2.circle(frame, (A[0], A[1]), 15, (255, 0, 0), -1)
            cv2.circle(frame, (B[0], B[1]), 15, (255, 0, 0), -1)
            cv2.circle(frame, (C[0], C[1]), 15, (255, 0, 0), -1)


          

                            # Calculate angle
            angle = calculate_angle(A, B, C)
            angle1= calculate_angle(D, E, F)
            angle2= calculate_angle(G, H, I)
            angle3= calculate_angle(J, K, L)

            LANDMARKS=[angle,angle1,angle2,angle3]


          

            # Display the angle on the frame
            cv2.putText(frame, f'Angle: {int(angle2)} degrees', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, f'Angle: {int(angle1)} degrees', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, f'Angle: {int(angle)} degrees', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, f'Angle: {int(angle3)} degrees', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        # Display the resulting frame
        cv2.imshow('Body Landmarks Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

open_camera()



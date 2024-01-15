import cv2 
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mphands = mp.solutions.hands

def process_hand_landmarks(image, hands_detector):
    image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mphands.HAND_CONNECTIONS
            )

    return image_bgr
    

def main():
    cap = cv2.VideoCapture(0)
    hands = mphands.Hands()

    
    while True:
        ret, image = cap.read()
        if not ret:
            print("Failed to capture video frame")
            break

        
        processed_image = process_hand_landmarks(image, hands)

        cv2.imshow('Handtracker', processed_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

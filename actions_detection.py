# This is optional, you can detect hands using Mediapipe if you want
import mediapipe as mp

mp_hands = mp.solutions.hands.Hands

def detect_actions(hands_landmarks):
    actions = []
    if hands_landmarks:
        for hand in hands_landmarks:
            if max([lm.y for lm in hand.landmark]) > 0.7:
                actions.append("Writing / Cheat Slip")
    return actions

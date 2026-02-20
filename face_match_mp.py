import numpy as np

def match_face(face_encoding, known_encodings, known_details, tolerance=3000):
    if len(known_encodings) == 0:
        return None

    distances = np.linalg.norm(np.array(known_encodings) - np.array(face_encoding), axis=1)
    best_index = np.argmin(distances)

    if distances[best_index] < tolerance:
        return known_details[best_index]
    return None

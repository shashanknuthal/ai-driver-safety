from features.face_processor import eye_aspect_ratio, mouth_aspect_ratio


def test_eye_ratio_positive():
    points = [(0,1),(1,0),(2,0),(4,1),(2,2),(1,2)]
    assert eye_aspect_ratio(points) > 0


def test_mouth_ratio_positive():
    points = [(0,1),(4,1),(2,0),(2,3),(1,1),(3,2)]
    assert mouth_aspect_ratio(points) > 0

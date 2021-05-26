class STObject():
    def __init__(self, type, topLeftX, topLeftY, bottomRightX, bottomRightY):
        self.type = type
        self.x1 = topLeftX
        self.y1 = topLeftY
        self.x2 = bottomRightX
        self.y2 = bottomRightY
        self.cx = (topLeftX + bottomRightX) / 2.0  # centroid
        self.cy = (topLeftY + bottomRightY) / 2.0


if __name__ == '__main__':
    o1 = STObject('something', 1, 2, 3, 4)
    o2 = STObject('something else', 2, 3, 4, 5)
    print(o1.type)
    print(o1.cx)
    print(o2.cy)

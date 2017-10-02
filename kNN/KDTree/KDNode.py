class KDNode:

    # if a node, then point is None. Otherwise, if a leaf, all but not point are None
    def __init__(self, point, med, div, left, right):
        self.point = point
        self.med = med
        self.div = div
        self.left = left
        self.right = right
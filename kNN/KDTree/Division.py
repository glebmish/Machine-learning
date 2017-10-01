class Division:

    x = "X"
    y = "Y"

    @staticmethod
    def switch_division(div):
        if div == Division.x:
            return Division.y
        else:
            return Division.x
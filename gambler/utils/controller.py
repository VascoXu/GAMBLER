class Controller:
    def __init__(self, 
                 alpha: float,
                 P: float,
                 setpoint: float):

        self._P = P         
        self._alpha = alpha 

        self._setpoint = setpoint

    def step(self, feedback, previous):
        error = self._setpoint - feedback
        stability = (1-self._P)
        return min(max(previous - ((stability/ALPHA) * error), 0.04722215648553174), 0.9893930903820092)

    def set_alpha(self, alpha: float):
        self._alpha = alpha
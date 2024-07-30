import chex

@chex.dataclass
class GoKartGeometry:
  l1: float = 0.72 # Distance from cog to front tires
  l2: float = 0.47 # Distance from cog to rear tires
  w1: float = 0.94 # Distance between front Tires
  w2: float = 1.08 # Distance between rear Tires
  back2backaxle: float = 0.23 # Distance from the rear of the gokart to the back axle
  frontaxle2front: float = 0.33 # Distance from the front axle to the front of the kart
  wheel2border: float = 0.18 # Side distance between center of the wheel and external frame
  F2n: float = l1/(l1 + l2) # Normal force at the rear axle "portion of Mass supported by rear tire"

@chex.dataclass
class TricycleParams:
  Iz: float = 0.7 # Inertia around the z axis
  REG_: float = 0.5 # Regularization factor

@chex.dataclass  
class PajieckaParams:
  class front_paj:
    B: float = 17.17
    C: float = 1.26
    D: float = 0.8
    E: float = 0.42
  class rear_paj:
    B: float = 13.02
    C: float = 1.27
    D: float = 0.97
    E: float = 0.21
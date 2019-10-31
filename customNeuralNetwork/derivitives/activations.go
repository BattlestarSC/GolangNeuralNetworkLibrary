package derivitives

import ("math")

/*
  Derivitives of activation functions
  -> Options: Sigmoid (1), Relu (2), Leaky Relu (3), Relu-6 (4), Custom (bi-leaky Relu-6) (5)
  output float64 for backprop
*/

func ActivationDerivities(function int, value float32) float64 {
  switch function {
  case 1:
    return sigmoid_der(value)
  case 2:
    return relu_der(value)
  case 3:
    return leaky_der(value)
  case 4:
    return relu_6_der(value)
  default: //default case 5
    return custom_der(value)
  }
}

//the sigmoid function 1 / 1 + e^-x
func sigmoid (x float32) float64 {
  eTo    := math.Exp(float64(-x))
  bottom := float64(1) + eTo
  all    := float64(1) / bottom
  return all
}

//defined as sigmoid * (1 - sigmoid)
func sigmoid_der(x float32) float64 {
  sig := sigmoid(x)
  oth := float64(1) - sig
  out := sig * oth
  return out
}

//relu derivitive
//0 if x <= 0
//1 otherwise
func relu_der(x float32) float64 {
  if x > 0 {
    return float64(1)
  } else {
    return float64(0)
  }
}

//leaky relu
//0.0001 if x <= 0
//1 if x > 0
func leaky_der(x float32) float64 {
  if x > 0 {
    return float64(1)
  } else {
    return float64(0.0001)
  }
}

//relu-6
//if x > 0 && x < 6, 1
//else, 0
func relu_6_der(x float32) float64 {
  if x > 0 && x < 6 {
    return float64(1)
  } else {
    return float64(0)
  }
}

//custom derivitive is same as relu 6, but if x < 0, 0.001, if x > 6, 0.0001
func custom_der(x float32) float64 {
  if x < 0 {
    return float64(0.001)
  } else if x > 6 {
    return float64(0.0001)
  } else {
    return float64(1)
  }
}

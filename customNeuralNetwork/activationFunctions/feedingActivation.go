package activationFunctions

import "math"

/*
 This function will provide activation functions, referenced by function (int) number
 -> Options: Sigmoid (1), Relu (2), Leaky Relu (3), Relu-6 (4), Custom (bi-leaky Relu-6) (5)
*/

func Activation(input float32, function int) float32 {

  if function == 1 { //If we are to activate with the sigmoid
    return sigmoid(input)
  } else if function == 2 { //If we are to activate with the Relu
    return relu(input)
  } else if function == 3 { //If we are to activate with the leaky relu
    return leaky(input)
  } else if function == 4 { //If we are to activate with the relu 6
    return r6(input)
  } else { //default to the custom activation function
    return customActivation(input)
  }

}

func sigmoid(x float32) float32 {
  //First e^-x
  first  := float32( math.Exp( float64(-x)  )  )
  //Bottom 1 + e^-x
  bottom := float32(1) + first
  //entire 1 / (1 + e^-x)
  entire := float32(1) / bottom

  return entire
}

func relu(x float32) float32 {
  if x <= float32(0) {
    return float32(0)
  } else {
    return x
  }
}

func leaky(x float32) float32 {
  if x <= float32(0) {
    reduced := x * float32(0.0001)
    return reduced
  } else {
    return x
  }
}

func r6(x float32) float32 {
  if x <= float32(0) {
    return float32(0)
  } else if x > float32(6) {
    return float32(6)
  } else {
    return x
  }
}

/*
  I'm playing with a custom bi-leaky relu6 function such that
  --when x <= 0; Output = 0.001x
  --when x > 0 and x < 6; Output = x
  --when x > 6; Output = 0.0001x
  My thought is that by allowing spikes in the network by not capping at 6 and not overly training/erroring high spikes, this will permit the network to learn spikes slowly
  The goal of this is to hopefully allow the network, when training without turning nodes off and on, with additional training itterations, will permit learning highly sparce data/learn small correlations well
  *Kinda just a personal thought experminent
*/

func customActivation(x float32) float32 {
  if x <= float32(0) {
    reduced := x * float32(0.001)
    return reduced
  } else if x > float32(6) {
    reduced := x * float32(0.0001)
    return reduced //+ float32(6) - For some reason, making this differentiable/smooth causes learning to just die and repeat
  } else {
    return x
  }
}

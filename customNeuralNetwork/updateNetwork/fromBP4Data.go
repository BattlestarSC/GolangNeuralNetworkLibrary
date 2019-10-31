package updateNetwork

/*
  This will take the output from organization and update all the weights and biases (optional) in the network
*/

import (
  n "customNeuralNetwork/nnetwork"
  b "customNeuralNetwork/bp4"
)

func UpdateNetwork(network n.Network, weightErrors []b.WeightError, biasErrors []b.BiasError) {
  //first handle biases
  //if the network has them disabled, just skip them. Yeah, calculating them is a waste of CPU cycles, however because of the structure of the network being mutiable in nature, we will calculate them in case it is enabled during runtime
  if network.Bias {
    //for each value in the bias error
    for _,error := range biasErrors {
      //bias error has location information, so add/update on that data and trust it (kinda unsafe, but that's a problem for me 5 hours from now)
      network.Layers[error.LayerIndex].Nodes[error.NodeIndex].Bias -= float32(error.Error)
    } //end foreach bias error

  }//end if biases are enabled

  //now the same for weight errors, but no conditional
  for _,error := range weightErrors {
    //use the structure's positioning info and add/update the error
    network.Layers[error.LayerIndex].Nodes[error.NodeIndex].Weights[error.WeightIndex] -= float32(error.Error)
  } //end for each weight error

}//end UpdateNetwork

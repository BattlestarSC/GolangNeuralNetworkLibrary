package training

/*
  This will just do all of the singles for us, single Epoch
*/

import (
  n "customNeuralNetwork/nnetwork"
  b "customNeuralNetwork/bp4"
  o "customNeuralNetwork/organization"
  u "customNeuralNetwork/updateNetwork"
  //"math"
)

//return average cost, network output channel
func TrainSupervised(network n.Network, inputs, expected [][]float32, allowDead, returnNetworkOutput bool, learningRate float64) (float64, chan n.NetworkOutput) {
  //make sure the sizing of the inputs is correct
  if len(inputs) != len(expected) {panic("Incomplete run data provided")}
  for ind,_ := range inputs {
    if len(inputs[ind]) != network.NumInputs || len(expected[ind]) != network.NumOutputs {panic("Incomplete example data provided")}
  }

  //make the channels
  costChannel    := make(chan float64)
  weightChannel := make(chan b.WeightError)
  biasChannel   := make(chan b.BiasError)
  var networkOutput chan n.NetworkOutput
  if returnNetworkOutput {
    networkOutput = make(chan n.NetworkOutput)
  } else {
    networkOutput = nil
  }

  //pass the examples
  for index,_ := range inputs {
    go TrainOnceSupervised(network, inputs[index], expected[index], allowDead, returnNetworkOutput, weightChannel, biasChannel, costChannel, networkOutput)
  }

  //do the error averaging
  weightError, biasError := o.ErrorForNetwork(network, learningRate, len(inputs), weightChannel, biasChannel)

  //update the network
  u.UpdateNetwork(network, weightError, biasError)

  //average cost
  var avg float64
  var count int
  for i:=0;i<len(inputs);i++ {
    newValue := <-costChannel
    //if math.IsNaN(newValue) {panic("NaN cost encountered")}
    avg += newValue
    count++
  }
  avg = avg / float64(count)

  //return
  return avg, networkOutput

}

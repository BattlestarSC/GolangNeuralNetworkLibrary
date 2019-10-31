package training

/*
  This single training function will run a SINGLE example and return the channels of Weight and Bias errors
  The goal of doing a single WITHOUT updating is to make running n examples a concurrent process with goroutines
*/

import (
  b1 "customNeuralNetwork/bp1"
  b2 "customNeuralNetwork/bp2"
  b4 "customNeuralNetwork/bp4"
  n  "customNeuralNetwork/nnetwork"
  c  "customNeuralNetwork/costs"
)

//run a single supervised example and leave the pipes full
//This requires an expected/known output
func TrainOnceSupervised(network n.Network, inputs, expected []float32, allowDead, returnNetworkOutput bool, weightChannel chan b4.WeightError, biasChannel chan b4.BiasError, costChannel chan float64, networkOutputChan chan n.NetworkOutput) {
  //first check sanity
  if network.NumInputs  != len(inputs)   {panic("Wrong size input passed to network")}
  if network.NumOutputs != len(expected) {panic("Wrong size of expected output passed to network")}

  //then run the training example
  netout := n.FeedNetwork(network, inputs, allowDead)
  //get the output error
  outputError := b1.OutputError(expected, netout)
  //get the cost
  outputCost := c.MeansSquared(netout.Outputs, expected)
  //now do the rest of the BP; BP2 launches BP4 in a non-blocking manner and sends those outputs to the channels
  b2.BP2(network, netout, outputError, weightChannel, biasChannel)
  //send the cost to the cost channel
  costChannel <- outputCost
  //if asked, send the network output
  if returnNetworkOutput{
    networkOutputChan <- netout
  }
}

package nodes

import s "strings"
import r "math/rand"

/*
  This function will create a node structure based off the provided parameters
  Index:              What node is this within the layer ---------------------------------------------------------------------------------- | Node->Index
  LastLayerNodes:     If connectionsList == Nil, this is equal to the length of the randomly generated weights ---------------------------- | Node->LastLayerNodes, Node->Weights (opt)
  FullyConnected:     If false, expect connectionsList != Nil, if true, ignore connectionsList, used to create a fully connected network -- | Node->Connections, Node->Weights (opt)
  Bias:               If true, create a bias in the network, if false, make a 0 bias ------------------------------------------------------ | Node->Bias
  ConnectionsList:    If not fully connected, this is a list of what nodes are connected from the last layer, and len(self) == len(Weights) | Node->Connections, Node->Weights (opt)
  ActivationFunction: Read from a list of activation functions and assign an integer for which to use ------------------------------------- | Node->Activation
						-> Options: Sigmoid (1), Relu (2), Leaky Relu (3), Relu-6 (4), Custom (bi-leaky Relu-6) (5)
*/

func MakeNode(index, lay, lastLayerNodes int, fullyConnected, bias bool, connectionsList []int, activationFunction string) Node {
  //Quick error checking
  if !fullyConnected && connectionsList == nil {panic("Node was not given enough information to exist")}

  //create the output variable
  var output Node

  //Set the easy stuff first
  output.Index          = index //Set the node index
  output.LastLayerNodes = lastLayerNodes //Set the last layer nodes
  output.Activation     = stringToActivationInt(activationFunction) //set the activation int
  output.Lay            = lay

  //Bias handling
  if bias { //if we use a bias
    output.Bias = r.Float32()
  } else { //if we do not use a bias
    output.Bias = float32(0)
  }

  //Handle if the network is or is not fully connected
  if fullyConnected {
  //If the network is fully connected

  //and set the inputs to the same as last layer nodes
  output.NumInputs = lastLayerNodes
  //Set connections to nil
  output.Connections = nil

  //create weights
  for i:=0;i<lastLayerNodes;i++ {
    output.Weights = append(output.Weights, r.Float32())
  }

  } else {
    //If the network is not fully connected

    //Set the output length
    output.NumInputs = len(connectionsList)
    //Save the list (should not be reused elsewhere in the program)
    output.Connections = connectionsList

    //create weights
    for i:=0;i<output.NumInputs;i++ {
      output.Weights = append(output.Weights, r.Float32())
    }
  }

  //number of weights
  if fullyConnected {
    output.NumWeights = lastLayerNodes
  } else {
    output.NumWeights = output.NumInputs
  }

  return output

}

/*
  This function provides the translation for activation functions to int
  -> Options: Sigmoid (1), Relu (2), Leaky Relu (3), Relu-6 (4), Custom (bi-leaky Relu-6) (5)
*/

func stringToActivationInt(input string) int {
  //large if structure
 if  s.Contains(s.ToLower(input), "sigmoid") { //check for sigmoid; if it contains "sigmoid"
   return 1
 } else if s.Contains(s.ToLower(input), "relu") && !s.Contains(s.ToLower(input), "leaky") && !s.Contains(s.ToLower(input), "6") { //relu if it matches relu, and does not contain either leaky or 6
   return 2
 } else if s.Contains(s.ToLower(input), "relu") && s.Contains(s.ToLower(input), "leaky") { //leaky relu if matches relu and leaky
   return 3
 } else if s.Contains(s.ToLower(input), "relu") && s.Contains(s.ToLower(input), "6") { //relu 6 if matches relu and 6
   return 4
 } else if s.Contains(s.ToLower(input), "custom") { //custom if matches custom
   return 5
 } else { //otherwise, I want what the user has been smoking
   panic("Node specified with inproper activation function string\nRecongize sigmoid, relu, leaky relu, relu 6, custom; Given: " + input + "\n")
 }

}

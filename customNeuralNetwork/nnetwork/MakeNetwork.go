package nnetwork

import (
  l  "customNeuralNetwork/layers"
)

/*
This is the set of functions for creating a network
This will create the hidden layers as specified, however the first hidden layer will take numInputs as its number of inputs
The numOutputs is a number of nodes for an output layer to be tacked on the end of the hidden layers
Thus, the number of layers is len(hiddenLayers) + 1 outputlayer
Admittitly, this does make it very weird to work with the library, however it seems to be easier than otherwise
The network will require at least one hidden layer and one output layer. In this 2 layer network, the hidden layer will act as an input AND hidden layer
*/

func MakeNetwork(numInputs, numOutputs int, hiddenLayers []int, fullyConnected, bias, multipleActivations bool, mainActivation string, activationList []string, connections [][][]int) Network {
  //first check for proper inputs -_-
  if hiddenLayers == nil {panic("No layers requested") } //if no layers given
  if !fullyConnected && connections == nil {panic("Unknown connections for partially connected network") } //if not fully connected but no connections provided
  if !fullyConnected && len(connections) != len(hiddenLayers)+1 {panic("Wrong number of partially connected connections provided") } //if not fully connected but the wrong number of connection lists provided
  if multipleActivations && activationList == nil  {panic("Not enough activations provided") } //if not singularly activated but other options not provided
  if multipleActivations && len(activationList) != len(hiddenLayers)+1 {panic("Confused by the wrong number of activations provided") } //If the wrong number of activations provided

  //Now we can actually start

  //add the output layer to the list
  layerslist := append(hiddenLayers, numOutputs)

  //make the output variable
  var output Network

  //load the easy stuff
  output.NumInputs  = numInputs
  output.NumLayers  = len(layerslist)
  output.NumOutputs = numOutputs
  output.FullyConnected = fullyConnected

  //make the layers
  for index,value := range layerslist { //for each layer to be created

		//connections handling
		var cons [][]int
		if fullyConnected {
			cons = nil
		} else {
			cons = connections[index]
		}

		//activation handling
		var act string
		if multipleActivations {
			act = activationList[index]
		} else {
			act = mainActivation
		}

		//last layer handling
		var deaths_allowed bool
		if index == output.NumLayers - 1 {
			deaths_allowed = false
		} else {
			deaths_allowed = true
		}

		//last number of nodes (if its not the first layer, its layerlist's previous index, otherwise, its the input numbers)
		var lastNodes int
		if index > 0 {
			lastNodes = layerslist[index - 1]
		} else {
			lastNodes = numInputs
		}

		//actually make the layers
    newLayer := l.MakeLayer(index, lastNodes, value, fullyConnected, bias, deaths_allowed, act, cons)

    //save the new layer
    output.Layers = append(output.Layers, newLayer)
	}

  output.LayerListing = layerslist //FOR DEBUG ONLY
  output.Bias = bias //for BP

	return output

}

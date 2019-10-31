package layers

/*
  This will create a layer, and create nodes within it
*/

import (
  n "customNeuralNetwork/nodes"
)

func MakeLayer(index, lastNodes, numNodes int, fullyConnected, bias, lastLayer bool, activation string, connectionsList [][]int) Layer {
  //First some very basic error checking
  if !fullyConnected && connectionsList == nil {panic("Insufficient information to create a sane layer")}

  //Then make the output variable
  var output Layer

  //load the easy stuff first
  output.NumInputs = lastNodes
  output.Index = index
  output.LastLayer = lastLayer
  output.NumOutputs = numNodes

  //Make the nodes
  for i:=0;i<numNodes;i++{
    //select the list of connections to use
    var connectiondata []int
    if !fullyConnected {
      connectiondata = connectionsList[i]
    } else {
      connectiondata = nil
    }
    //make the node
    newNode := n.MakeNode(i, index, lastNodes, fullyConnected, bias, connectiondata, activation)

    //Add the new node to the layer
    output.Nodes = append(output.Nodes, newNode)
  }

  //rely on the nodes to resolve the int for the activation function
  output.Activation = output.Nodes[0].Activation

  //return the new layer
  return output
}

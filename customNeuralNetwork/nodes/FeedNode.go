package nodes

import (
  f "customNeuralNetwork/activationFunctions"
  "strconv"
  r "math/rand"
)

/*
  This function is to feed data though a node and return output though a channel ONLY
  Needs to function for any proper input
  allow_deaths is for options killing randomly of nodes during training, false for output layer and getting a network prediction
  *
  Reason this is channel output only: This permits simple concurency whenever feeding, so for a large load is being fed, all of these can run in parallel on a larger cpu, and it takes less time to launch all these go routines and then sort their outputs
*/

func FeedNode(n Node, inputs []float32, allow_deaths bool, outputChannel chan NodeOutput) {
  //fast error checking
  if n.LastLayerNodes != len(inputs) {panic(("Node was passed " + strconv.Itoa(len(inputs)) + " instead of " + strconv.Itoa(n.LastLayerNodes) + " inputs") )}

  //create the output variable
  var output NodeOutput
  //save index
  output.Index = n.Index

  //Copy inputs (make sure that inputs are never edited)
  output.Inputs = inputs
  output.Lay    = n.Lay

  //Check for the death
  if allow_deaths && r.Intn(125) == 7 {  //DEATH
    output.Output = float32(0)
    output.Pre    = float32(0)
    output.Dead   = true

  } else if n.Connections == nil {  //FULLY CONNECTED
    //Now handle passing a fully connected node
    var run float32 //The running sum for the weighted sum/pre-activation value
    for i,value := range n.Weights { //for every weight, multiply by the input (Trust the inputs, because during creation of the node, n.LastLayerNodes should equal number of inputs so error checking already happened)
      temp := value * inputs[i] //do the math
      run = run + temp //update
    } //end for each weight
    //Add bias
    run = run + n.Bias

    //compute activation
    final := f.Activation(run, n.Activation)

    //load output
    output.Output = final
    output.Pre = run
    output.Dead = false
    output.NumWeights = n.NumWeights


  } else { //PARTIALLY CONNECTED

    //Do the same thing as the fully connected network, but use only values from n.Connections
    var run float32
    for index,value := range n.Connections {
      //So if there are 5 connections, there will be 5 weights, so use n.Weights[index] and inputs[value]
      temp := n.Weights[index] * inputs[value]
      run = run + temp

    } //end foreach
    //Add bias
    run = run + n.Bias

    //Activation
    final := f.Activation(run, n.Activation)

    //Load output
    output.Output = final
    output.Pre = run
    output.Dead = false

    //load used Inputs
    output.Inputs = nil
    for _,value := range n.Connections {
      output.Inputs = append(output.Inputs, inputs[value])
    }

  }

  outputChannel <- output

}

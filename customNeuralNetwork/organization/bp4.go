package organization

import(
  n "customNeuralNetwork/nnetwork"
  b "customNeuralNetwork/bp4"
)

/*
  This will permit getting an average for every weight and bias from all of the outputs from BP4
*/

/*
  This will use a Network struture to count the total number of weights and baises in the network
  Number of nodes is the number of biases (even if biases are not applicable)
*/
func numWeightsAndBiases(input n.Network) (int, int) {
  //running total vars
  var runningWeights int
  var runningBiases  int

  //now for each layer in network
  for _,layer := range input.Layers {
    //the number of nodes is equal to the number of biases
    runningBiases = runningBiases + layer.NumOutputs
    //for each node, add its number of weights
    for _,nde := range layer.Nodes {
      //the node saves the needed data in NumWeights
      runningWeights = runningWeights + nde.NumWeights
    }
  } //end for each layer output in the input network output
  return runningWeights,runningBiases
}

/*
CLEAN THE PIPES!
This function will remove everything from the weight error and bias error channel and return it in a slice to another organizing function
Multiply lengths by number of runs
*/
func getWeightErrorsAndBiasErrors(numWeights, numBiases, numRuns int, weightChannel chan b.WeightError, biasChannel chan b.BiasError) ([]b.WeightError, []b.BiasError) {
  //get weights from channel
  var weightOutputs []b.WeightError
  //for each weight in ALL runs, append it from the channel to the list
  for i:=0;i<(numWeights * numRuns);i++{
    weightOutputs = append(weightOutputs, <- weightChannel)
  }
  //and now the same for biases
  var biasOutputs []b.BiasError
  for i:=0;i<(numBiases * numRuns);i++ {
    biasOutputs = append(biasOutputs, <- biasChannel)
  }

  return weightOutputs, biasOutputs
}

/*
  Average weight error for a node's weight, given EVERY error, so sort and count
  Return though a channel so many of these can run at once
*/
func getErrorForNodeWeight(layerIndex, nodeIndex, weightIndex int, input []b.WeightError, outputChannel chan b.WeightError){
  //alright, create an averaging variable and an counter, so no copying is required
  var average float64
  var number  int

  //for every weightError
  for _,weightError := range input {
    //match positioning, ignore deaths
    if weightError.LayerIndex == layerIndex && nodeIndex == weightError.NodeIndex && weightError.WeightIndex == weightIndex && weightError.Dead == false {
      average = average + weightError.Error
      number  = number + 1
    }
  }

  //divide for actual average
  average = average / float64(number)

  //make the new weightError structure
  var output b.WeightError
  output.LayerIndex  = layerIndex
  output.NodeIndex   = nodeIndex
  output.WeightIndex = weightIndex
  output.Error       = average
  output.Dead        = false //meaningless now );

  outputChannel <- output
}

/*
  Same as getErrorForNodeWeight but for biases
*/
func getErrorForNodeBias(layerIndex, nodeIndex int, input []b.BiasError, outputChannel chan b.BiasError) {
  //this is to average every matching bias error
  var average float64
  var count   int

  //for each in the slice
  for _,biasErr := range input {
    //match outputs except for deaths
    if biasErr.LayerIndex == layerIndex && biasErr.NodeIndex == nodeIndex && biasErr.Dead == false {
      average = average + biasErr.Error
      count = count + 1
    }
  }

  //divide for average
  average = average / float64(count)

  //make output variable out of a biasError structure
  var output b.BiasError
  output.LayerIndex = layerIndex
  output.NodeIndex  = nodeIndex
  output.Error      = average
  output.Dead       = false //meaingless now

  outputChannel <- output
}

/*
  Now integrate them all
*/
func ErrorForNetwork(network n.Network, learningRate float64, numberOfRuns int, weightChannel chan b.WeightError, biasChannel chan b.BiasError) ([]b.WeightError, []b.BiasError) {
  //first get sizing info
  numberOfWeights, numberOfBiases := numWeightsAndBiases(network)
  //then make channels for threaded error averaging
  internalWeightChannel := make(chan b.WeightError)
  internalBiasChannel   := make(chan b.BiasError)
  //Get all of the errors from the channels (clean the pipes)
  weightErrors, biasError := getWeightErrorsAndBiasErrors(numberOfWeights,numberOfBiases,numberOfRuns,weightChannel,biasChannel)
  //Get sizing information, lists of indexes to go through
  //rather, just go through each layer in the network
  for layerIndex,layer := range network.Layers {
    //then go through each node in the layer
    for nodeIndex,node := range layer.Nodes {
      //here we can do the bias error
      go getErrorForNodeBias(layerIndex, nodeIndex, biasError, internalBiasChannel)
      //and go though the length of node weights
      for weightIndex:=0;weightIndex<node.NumWeights;weightIndex++{
        //for each weight, launch
        go getErrorForNodeWeight(layerIndex, nodeIndex, weightIndex, weightErrors, internalWeightChannel)
      }
    }
  }

  //then get them all in a slice and multiply by learning rate
  var avgWeightErrors []b.WeightError
  var avgBiasErrors   []b.BiasError
  //collect weight errors
  for i:=0;i<numberOfWeights;i++{
    //collect
    temp := <- internalWeightChannel
    //learning rate multiply
    temp.Error = temp.Error * learningRate
    //append
    avgWeightErrors = append(avgWeightErrors, temp)
  }
  //collect bias errors
  for i:=0;i<numberOfBiases;i++{
    //collect
    temp := <- internalBiasChannel
    //learning rate
    temp.Error = temp.Error * learningRate
    //append
    avgBiasErrors = append(avgBiasErrors, temp)
  }
  //return
  return avgWeightErrors, avgBiasErrors
}

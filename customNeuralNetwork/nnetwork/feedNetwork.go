package nnetwork

import (
  l "customNeuralNetwork/layers"
)

/*
  This is all the functions required to feed data through a network for either training or prediction
*/

func FeedNetwork(n Network, inputs []float32, training bool) NetworkOutput {

	//first check input sanity
	if n.NumInputs != len(inputs) {panic("Wrong number of inputs provided to feeding a network") }

	//variable for last output
	var lastOutput []float32
	//set it to inputs in prep for feeding
	lastOutput = inputs

	//variable to acumulate layer outputs
	var runningData []l.LayerOutput

	//run though the layers
	for _,layer := range n.Layers {
		//temporary save
		var temp l.LayerOutput

		//run the layer
		temp = l.FeedLayer(layer, lastOutput, training)

		//save the data
		lastOutput = temp.Output
		runningData = append(runningData, temp)
	}

	//Now load the output structure
	var output NetworkOutput

	//set the output first
	output.Outputs = lastOutput
	//Then save the rest
	output.LayerOutputs = runningData //This is not to be edited, so saving it this way is fine
	//Count the deaths
	var death_count int
	for _,value := range runningData { //for each layeroutput
		for _,answer := range value.DeadNodes { //for every node's output
			if answer == true {
				death_count = death_count + 1
			}
		} //end for each node output
	} //end for each layeroutput

	output.NumDead = death_count

	return output

}

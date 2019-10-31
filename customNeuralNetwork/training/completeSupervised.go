package training

import (
  "math/rand"
  n "customNeuralNetwork/nnetwork"
  "fmt"
  "time"
)

/*
  This is to do the complete training process, random subset of data, epochs, updating, and running
  This will rely on a random number generator to select subsets of data fairly
*/

//this function will create a list of indexes to use for data
func selectData(dataLength, targetLength int) []int {
  var output []int
  //now loop through options and randomly select till full
  for i:=0;len(output)<targetLength;i++ {
    //reset if running multiple times
    if i == dataLength {
      i = 0
    }    //end if too long

    if 1 == rand.Intn(10) {//10% chance of being picked
      output = append(output, i) //don't care about random repeats
    }//end if select

  }//end for loop

  return output

}

//launch and epoch TrainingSupervised
func TrainSupervisedNetwork(network n.Network, inputData, expectedOutputs [][]float32, examplesPerEpoch, numberOfEpochs int, learningRate float64) {
  //sanity check
  if len(inputData) != len(expectedOutputs) {panic("Incorrect data sizing")}
  //for each epoch
  for epoch:=0;epoch<numberOfEpochs;epoch++ {
    //start timeing
    epochStart := time.Now()
    //make training dataset
    var trainingSet, expectedSet [][]float32
    //which examples to use
    indicies := selectData(len(inputData), examplesPerEpoch)
    //now load the new sets
    for _,ind := range indicies { //for each index to use
      trainingSet = append(trainingSet, inputData[ind])
      expectedSet = append(expectedSet, expectedOutputs[ind])
    } //end for each index to use

    //now data has been selected, run/feed
    cost,_ := TrainSupervised(network, trainingSet, expectedSet, true, false, learningRate)
    fmt.Println("Epoch", epoch, "cost:", cost, "time elapsed:", time.Since(epochStart))
  } //end epoch

}

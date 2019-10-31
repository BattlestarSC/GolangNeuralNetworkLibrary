package costs

import (
	"math"
)

/*
	Means squared error for a network output
	Takes n actual outputs and n expected outputs and does means squared error
*/

func MeansSquared(actual, expected []float32) float64 {
	//if not the same sizes, break
	if len(actual) != len(expected) {panic("Malformed inputs to meanssquared error")}

	//keep track of the summation
	var runningSummation float64

	//for each input
	for i,_ := range actual {
		//ABS actual - expected
		p1 := actual[i] - expected[i]
		p2 := math.Abs(float64(p1))
		p3 := p2 * p2
		runningSummation = runningSummation + p3
	}

	//divide by the number of inputs * 2 (because one half MSE is a stats thing)
	output := runningSummation / float64(len(actual) * 2)

	return output


}


//derivitive of mse, ends up just being y - y' per node
func MSEDerPerNode(actual, expected []float32) []float64 {
	if len(actual) != len(expected) {panic("Malformed inputs to meanssquared error derivitive")}

	//keep track of output
	var output []float64
	for i,_ := range actual {
		output = append(output, ((float64(expected[i] - actual[i])*float64(-1))))
	}

	return output

}

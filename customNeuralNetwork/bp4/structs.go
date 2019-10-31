package bp4

type WeightError struct {
  LayerIndex  int
  NodeIndex   int
  WeightIndex int
  Error       float64
  Dead        bool
}

type BiasError struct {
  LayerIndex int
  NodeIndex  int
  Error      float64
  Dead       bool
}

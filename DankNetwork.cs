using System;
using System.Collections.Generic;
using DankNN.Activation;
using DankNN.Error;
using DankNN.Layers;

namespace DankNN
{
    public class DankNetwork
    {
        public readonly DankLayer inputLayer;
        public readonly List<DankConnectedLayer> connectedLayers = new List<DankConnectedLayer>();

        public DankLayer EndLayer => connectedLayers.Count > 0 ? connectedLayers[^1] : inputLayer;
        public DankConnectedLayer OutputLayer => connectedLayers[^1];

        public DankNetwork(int numInputNeurons)
        {
            inputLayer = new DankLayer(numInputNeurons);
        }

        private static Random random = new Random();
        private static double GenerateNormalRandom()
        {
            return Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) *
                   Math.Sin(2.0 * Math.PI * (1.0 - random.NextDouble()));
        }

        public void ApplyNormalInit()
        {
            foreach (DankConnectedLayer layer in connectedLayers)
                unsafe
                {
                    fixed (double* pConnectionWeights = layer.connectionWeights)
                    {
                        for (var i = 0; i < layer.connectionWeights.Length; i++)
                            pConnectionWeights[i] = GenerateNormalRandom();
                    }
                }
        }

        private const double SqrtTwo = 1.41421356237;
        public void ApplyKaiserInit()
        {
            foreach (DankConnectedLayer layer in connectedLayers)
            {
                var neuronCount = layer.connectionWeights.GetLength(0);
                var kaiserVal = SqrtTwo / Math.Sqrt(neuronCount);

                unsafe
                {
                    fixed (double* pConnectionWeights = layer.connectionWeights)
                    {
                        for (var i = 0; i < layer.connectionWeights.Length; i++)
                            pConnectionWeights[i] = GenerateNormalRandom() * kaiserVal;
                    }
                }
            }
        }

        public DankConnectedLayer MakeLayer(int numNeurons, IDankActivation activation)
        {
            var layer = new DankConnectedLayer(numNeurons, EndLayer, activation);
            connectedLayers.Add(layer);

            return layer;
        }

        public DankLayer MakeLayer(int numNeurons)
        {
            return MakeLayer(numNeurons, DankLinearActivation.Singleton);
        }

        public double[] Propagate(double[] inputs)
        {
            if (inputs.Length != inputLayer.neurons.Length)
                throw new ArgumentException(
                    $"The length of {nameof(inputs)} must be the same as the number of input neurons", nameof(inputs));

            // Set the first layer's values
            for (var i = 0; i < inputLayer.neurons.Length; i++) inputLayer.neurons[i] = inputs[i];

            for (var i = 1; i < connectedLayers.Count; i++) connectedLayers[i].Forward();

            return OutputLayer.neurons;
        }

        public void Backpropagate(double[] errorDerivatives)
        {
            var outputLayer = OutputLayer;

            if (errorDerivatives.Length != outputLayer.neurons.Length)
                throw new ArgumentException(
                    $"The length of {nameof(errorDerivatives)} must be the same as the number of output neurons",
                    nameof(errorDerivatives));

            outputLayer.AddError(errorDerivatives);

            for (var i = connectedLayers.Count - 2; i > 0; i--) connectedLayers[i].Backward();
        }

        public double Backpropagate(double[] expectedOutputs, IDankError error)
        {
            DankLayer outputLayer = OutputLayer;

            if (expectedOutputs.Length != outputLayer.neurons.Length)
                throw new ArgumentException(
                    $"The length of {nameof(expectedOutputs)} must be the same as the number of output neurons",
                    nameof(expectedOutputs));

            // Set the last layer's error values and calculate loss
            double loss = 0;
            var errorDerivatives = new double[expectedOutputs.Length];
            for (var i = 0; i < outputLayer.neurons.Length; i++)
            {
                loss += error.Calculate(outputLayer.neurons[i], expectedOutputs[i]);
                errorDerivatives[i] = error.Derivative(outputLayer.neurons[i], expectedOutputs[i]);
            }

            Backpropagate(errorDerivatives);

            return loss;
        }

        public void ApplyErr(double learningRate = 1)
        {
            for (var i = 1; i < connectedLayers.Count; i++) connectedLayers[i].ApplyError(learningRate);
        }
    }
}
using System;
using System.Collections.Generic;
using System.IO;
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

        private static readonly Random InitRandom = new Random();
        private static double GenerateNormalRandom()
        {
            return Math.Sqrt(-2.0 * Math.Log(1.0 - InitRandom.NextDouble())) *
                   Math.Sin(2.0 * Math.PI * (1.0 - InitRandom.NextDouble()));
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

        public DankConnectedLayer MakeLayer(int numNeurons, IDankActivation activation, double dropout = 0.0)
        {
            var layer = new DankConnectedLayer(numNeurons, EndLayer, activation, dropout);
            connectedLayers.Add(layer);

            return layer;
        }

        public DankLayer MakeLayer(int numNeurons, double dropout = 0.0)
        {
            return MakeLayer(numNeurons, DankLinearActivation.Singleton, dropout);
        }

        public double[] Propagate(double[] inputs)
        {
            if (inputs.Length != inputLayer.neurons.Length)
                throw new ArgumentException(
                    $"The length of {nameof(inputs)} must be the same as the number of input neurons", nameof(inputs));

            // Set the first layer's values
            for (var i = 0; i < inputLayer.neurons.Length; i++) inputLayer.neurons[i] = inputs[i];

            foreach (var layer in connectedLayers)
                layer.Forward();

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

            for (var i = connectedLayers.Count - 2; i >= 0; i--) connectedLayers[i].Backward();
        }

        public double Backpropagate(double[] expectedOutputs, IDankError error)
        {
            DankLayer outputLayer = OutputLayer;

            if (expectedOutputs.Length != outputLayer.neurons.Length)
                throw new ArgumentException(
                    $"The length of {nameof(expectedOutputs)} must be the same as the number of output neurons",
                    nameof(expectedOutputs));

            // Calculate the last layer's error values and calculate loss
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

        public double CalculateLoss(double[] expectedOutputs, IDankError error)
        {
            DankLayer outputLayer = OutputLayer;

            if (expectedOutputs.Length != outputLayer.neurons.Length)
                throw new ArgumentException(
                    $"The length of {nameof(expectedOutputs)} must be the same as the number of output neurons",
                    nameof(expectedOutputs));

            // Calculate loss for each output
            double loss = 0;
            for (var i = 0; i < outputLayer.neurons.Length; i++)
                loss += error.Calculate(outputLayer.neurons[i], expectedOutputs[i]);

            return loss;
        }

        public void ApplyErr(double learningRate = 1)
        {
            foreach (var layer in connectedLayers)
                layer.ApplyError(learningRate);
        }

        public void Save(string path)
        {
            BinaryWriter writer = new BinaryWriter(File.Create(path));

            try
            {
                foreach (var layer in connectedLayers)
                {
                    unsafe
                    {
                        fixed (double* pConnectionWeights = layer.connectionWeights)
                        {
                            for (var i = 0; i < layer.connectionWeights.Length; i++)
                                writer.Write(pConnectionWeights[i]);
                        }
                    }

                    foreach (var neuronBias in layer.neuronBiases) writer.Write(neuronBias);
                }
            }
            finally
            {
                writer.Close();
            }
        }

        public void Load(string path)
        {
            BinaryReader reader = new BinaryReader(File.OpenRead(path));

            try
            {
                foreach (var layer in connectedLayers)
                {
                    unsafe
                    {
                        fixed (double* pConnectionWeights = layer.connectionWeights)
                        {
                            for (var i = 0; i < layer.connectionWeights.Length; i++)
                                pConnectionWeights[i] = reader.ReadDouble();
                        }
                    }

                    for (var i = 0; i < layer.neuronBiases.Length; i++) layer.neuronBiases[i] = reader.ReadDouble();
                }
            }
            finally
            {
                reader.Close();
            }
        }
    }
}
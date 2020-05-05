using System;
using DankNN.Activation;
using DankNN.Util;

namespace DankNN.Layers
{
    public class DankConnectedLayer : DankLayer
    {
        public readonly DankLayer parent;
        public IDankActivation activation;

        protected readonly double[] rawNeurons;

        public readonly double[] neuronBiases;
        public readonly double[] neuronErrors;

        public readonly double[,] connectionWeights;
        public readonly double[,] connectionErrors;

        public int numErrors = 0;

        public DankConnectedLayer(int numNeurons, DankLayer parent, IDankActivation activation) : base(numNeurons)
        {
            this.parent = parent;
            parent.child = this;

            this.activation = activation;

            rawNeurons = new double[numNeurons];

            neuronBiases = new double[numNeurons];
            neuronErrors = new double[numNeurons];

            connectionWeights = new double[numNeurons, parent.neurons.Length];
            connectionErrors = new double[numNeurons, parent.neurons.Length];
        }

        public void Forward()
        {
            connectionWeights.Times(parent.neurons, neurons);

            for (var i = 0; i < neurons.Length; i++)
            {
                var inputValue = neurons[i] + neuronBiases[i];

                rawNeurons[i] = inputValue;
                neurons[i] = activation.Calculate(inputValue);
            }
        }

        public void AddError(double[] errors)
        {
            // Pass the error to the neurons
            for (var i = 0; i < rawNeurons.Length; i++)
            {
                var neuronError = errors[i] * activation.Derivative(rawNeurons[i]);

                rawNeurons[i] = neuronError;
                neuronErrors[i] += neuronError;
            }

            // Pass the error through the connections
            unsafe
            {
                var height = parent.neurons.Length;
                var width = rawNeurons.Length;

                fixed (double* pNeuronError = rawNeurons, pParentNeurons = parent.neurons, pConnectionErrors =
                    connectionErrors)
                {
                    int offsetRow;
                    for (var row = 0; row < height; row++)
                    {
                        offsetRow = row * width;

                        for (var offset = 0; offset < width; offset++)
                            pConnectionErrors[offsetRow + offset] += pNeuronError[row] * pParentNeurons[offset];
                    }
                }
            }

            numErrors++;
        }

        public void Backward()
        {
            if (child == null)
                throw new NullReferenceException($"{nameof(child)} must be defined to propagate upstream");

            AddError(child.connectionWeights.TransposeTimes(child.rawNeurons));
        }

        public void ApplyError(double learningRate)
        {
            // Apply the error to the neuron biases
            for (var i = 0; i < neuronBiases.Length; i++)
            {
                neuronBiases[i] -= (learningRate * neuronErrors[i]) / numErrors;
                neuronErrors[i] = 0;
            }

            // Apply the error to the connection weights
            unsafe
            {
                fixed (double* pConnectionWeights = connectionWeights, pConnectionErrors = connectionErrors)
                {
                    for (var i = 0; i < connectionWeights.Length; i++)
                    {
                        pConnectionWeights[i] -= (learningRate * pConnectionErrors[i]) / numErrors;
                        pConnectionErrors[i] = 0;
                    }
                }
            }

            numErrors = 0;
        }
    }
}
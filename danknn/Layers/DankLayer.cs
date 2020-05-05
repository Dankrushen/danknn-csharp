using System;

namespace DankNN.Layers
{
    public class DankLayer
    {
        public readonly double[] neurons;

        public DankConnectedLayer? child;

        public DankLayer(int numNeurons)
        {
            if (numNeurons <= 0)
                throw new ArgumentOutOfRangeException(nameof(numNeurons), numNeurons,
                    $"{nameof(numNeurons)} must be greater than 0");

            neurons = new double[numNeurons];
        }
    }
}
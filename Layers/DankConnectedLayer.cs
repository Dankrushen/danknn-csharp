using DankNN.Activation;

namespace DankNN.Layers
{
    public class DankConnectedLayer : DankLayer
    {
        public DankLayer parent;
        public IDankActivation activation;

        public double[] neuronBiases;
        public double[] neuronErrors;

        public double[,] connectionWeights;
        public double[,] connectionErrors;

        public int numErrors = 0;

        public DankConnectedLayer(int numNeurons, DankLayer parent, IDankActivation activation) : base(numNeurons)
        {
            this.parent = parent;
            this.activation = activation;

            neuronBiases = new double[numNeurons];
            neuronErrors = new double[numNeurons];

            connectionWeights = new double[parent.neurons.Length, numNeurons];
            connectionErrors = new double[parent.neurons.Length, numNeurons];
        }

        public void Forward()
        {

        }

        public void Backward()
        {

        }

        public void ApplyError()
        {

        }
    }
}
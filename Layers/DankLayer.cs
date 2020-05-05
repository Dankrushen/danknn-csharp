namespace DankNN.Layers
{
    public class DankLayer
    {
        public readonly double[] neurons;

        public DankConnectedLayer? child;

        public DankLayer(int numNeurons)
        {
            neurons = new double[numNeurons];
        }
    }
}
namespace DankNN.Layers
{
    public class DankLayer
    {
        public double[] neurons;

        public DankConnectedLayer? child;

        public DankLayer(int numNeurons)
        {
            neurons = new double[numNeurons];
        }
    }
}
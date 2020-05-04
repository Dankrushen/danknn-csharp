namespace DankNN.Activation
{
    public class DankLeakyReluActivation : IDankActivation
    {
        public static readonly DankLeakyReluActivation Singleton = new DankLeakyReluActivation(0.01F);

        public readonly double leakRate;

        public DankLeakyReluActivation(double leakRate)
        {
            this.leakRate = leakRate;
        }

        public double Calculate(double val)
        {
            return val < 0 ? val * leakRate : val;
        }

        public double Derivative(double val)
        {
            return val < 0 ? leakRate : 1;
        }
    }
}
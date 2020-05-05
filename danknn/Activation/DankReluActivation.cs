namespace DankNN.Activation
{
    public class DankReluActivation : IDankActivation
    {
        public static readonly DankReluActivation Singleton = new DankReluActivation();

        public double Calculate(double val)
        {
            return val < 0 ? 0 : val;
        }

        public double Derivative(double val)
        {
            return val < 0 ? 0 : 1;
        }
    }
}
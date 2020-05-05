namespace DankNN.Activation
{
    public class DankLinearActivation : IDankActivation
    {
        public static readonly DankLinearActivation Singleton = new DankLinearActivation();

        public double Calculate(double val)
        {
            return val;
        }

        public double Derivative(double val)
        {
            return 1;
        }
    }
}
using System;

namespace DankNN.Activation
{
    public class DankSigmoidActivation : IDankActivation
    {
        public static readonly DankSigmoidActivation Singleton = new DankSigmoidActivation();

        public double Calculate(double val)
        {
            return 1 / (1 + Math.Exp(-val));
        }

        public double Derivative(double val)
        {
            var output = Calculate(val);
            return output * (1 - output);
        }
    }
}
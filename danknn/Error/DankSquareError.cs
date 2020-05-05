namespace DankNN.Error
{
    public class DankSquareError : IDankError
    {
        public static readonly DankSquareError Singleton = new DankSquareError();

        public double Calculate(double val, double target)
        {
            var error = val - target;
            return 0.5F * error * error;
        }

        public double Derivative(double val, double target)
        {
            return val - target;
        }
    }
}
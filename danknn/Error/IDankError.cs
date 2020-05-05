namespace DankNN.Error
{
    public interface IDankError
    {
        double Calculate(double val, double target);
        double Derivative(double val, double target);
    }
}
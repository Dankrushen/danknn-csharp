namespace DankNN.Activation
{
    public interface IDankActivation
    {
        double Calculate(double val);
        double Derivative(double val);
    }
}
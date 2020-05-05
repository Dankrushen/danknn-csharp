using System;
using DankNN.Activation;
using DankNN.Error;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DankNN.Tests
{
    [TestClass]
    public class DankNetworkTests
    {
        [TestMethod]
        public void DankNetworkTest()
        {
            var network = new DankNetwork(1);

            // Hidden layers
            network.MakeLayer(5, DankSigmoidActivation.Singleton);
            network.MakeLayer(5, DankSigmoidActivation.Singleton);
            network.MakeLayer(3, DankSigmoidActivation.Singleton);

            // Output
            network.MakeLayer(1, DankSigmoidActivation.Singleton);

            network.ApplyNormalInit();

            var random = new Random();
            for (var i = 0; i < 10000; i++)
            {
                var val = random.NextDouble() * 2.0 - 1.0;

                var predictions = network.Propagate(new[] {val});
                var loss = network.Backpropagate(val >= 0 ? new[] {1.0} : new[] {0.0}, DankSquareError.Singleton);

                if (i % 1000 == 0)
                    Console.WriteLine(
                        $"Iter: {i}, In: {val}, Out: {predictions[0]}, Loss: {loss}");

                network.ApplyErr();
            }
        }
    }
}
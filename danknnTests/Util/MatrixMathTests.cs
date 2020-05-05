using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DankNN.Util.Tests
{
    [TestClass]
    public class MatrixMathTests
    {
        [TestMethod]
        public void TimesTest()
        {
            var a = new double[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            var b = new double[,] {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
            var c = new double[,] {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}};

            var result = a.Times(b);

            for (var i = 0; i < c.GetLength(0); i++)
                for (var j = 0; j < c.GetLength(1); j++)
                    Assert.AreEqual(c[i, j], result[i, j]);
        }

        [TestMethod]
        public void TimesTest1()
        {
            var a = new double[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            var b = new double[] {9, 8, 7};
            var c = new double[] {46, 118, 190};

            var result = a.Times(b);

            for (var i = 0; i < c.Length; i++) Assert.AreEqual(c[i], result[i]);
        }

        [TestMethod]
        public void TransposeTimesTest()
        {
            var a = new double[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            var b = new double[] {9, 8, 7};
            var c = new double[] {90, 114, 138};

            var result = a.TransposeTimes(b);

            for (var i = 0; i < c.Length; i++) Assert.AreEqual(c[i], result[i]);
        }

        [TestMethod]
        public void PlusEqualsTest()
        {
            var a = new double[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            var b = new double[,] {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
            var c = new double[,] {{10, 10, 10}, {10, 10, 10}, {10, 10, 10}};

            a.PlusEquals(b);

            for (var i = 0; i < c.GetLength(0); i++)
                for (var j = 0; j < c.GetLength(1); j++)
                    Assert.AreEqual(c[i, j], a[i, j]);
        }
    }
}
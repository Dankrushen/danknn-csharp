namespace DankNN.Util
{
    public static class MatrixMath
    {
        public static double[,] Times(this double[,] a, double[,] b)
        {
            var height = a.GetLength(0);
            var width = b.GetLength(1);
            var length = a.GetLength(1);

            var result = new double[height, width];

            unsafe
            {
                fixed (double* pResult = result, pA = a, pB = b)
                {
                    int offsetRow, offsetColumn;
                    for (var row = 0; row < height; row++)
                    {
                        offsetRow = row * length;
                        for (var column = 0; column < width; column++)
                        {
                            offsetColumn = column;
                            double res = 0;
                            for (var offset = 0; offset < length; offset++, offsetColumn += width)
                                res += pA[offsetRow + offset] * pB[offsetColumn];
                            pResult[row * width + column] = res;
                        }
                    }
                }
            }

            return result;
        }

        public static double[] Times(this double[,] a, double[] b)
        {
            var height = a.GetLength(0);
            // The width is 1, therefore it can be ignored
            var length = a.GetLength(1);

            var result = new double[height];
            unsafe
            {
                fixed (double* pResult = result, pA = a, pB = b)
                {
                    int offsetRow;
                    for (var row = 0; row < height; row++)
                    {
                        offsetRow = row * length;

                        double res = 0;
                        for (var offset = 0; offset < length; offset++) res += pA[offsetRow + offset] * pB[offset];
                        pResult[row] = res;
                    }
                }
            }

            return result;
        }

        public static void Times(this double[,] a, double[] b, double[] result)
        {
            var height = a.GetLength(0);
            // The width is 1, therefore it can be ignored
            var length = a.GetLength(1);

            unsafe
            {
                fixed (double* pResult = result, pA = a, pB = b)
                {
                    int offsetRow;
                    for (var row = 0; row < height; row++)
                    {
                        offsetRow = row * length;

                        double res = 0;
                        for (var offset = 0; offset < length; offset++) res += pA[offsetRow + offset] * pB[offset];
                        pResult[row] = res;
                    }
                }
            }
        }

        public static double[] TimesHorizontal(this double[] a, double[,] b)
        {
            // The height is 1, therefore it can be ignored
            var width = b.GetLength(1);
            var length = a.Length;

            var result = new double[width];
            unsafe
            {
                fixed (double* pResult = result, pA = a, pB = b)
                {
                    int offsetColumn;
                    for (var column = 0; column < width; column++)
                    {
                        offsetColumn = column;
                        double res = 0;
                        for (var offset = 0; offset < length; offset++, offsetColumn += width)
                            res += pA[offset] * pB[offsetColumn];
                        pResult[column] = res;
                    }
                }
            }

            return result;
        }

        public static void PlusEquals(this double[,] a, double[,] b)
        {
            var length = a.Length;

            unsafe
            {
                fixed (double* pA = a, pB = b)
                {
                    for (var i = 0; i < length; i++) pA[i] += pB[i];
                }
            }
        }
    }
}
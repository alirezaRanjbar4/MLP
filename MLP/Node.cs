namespace MLP
{
    public class Node
    {
        public double Value { get; set; }
        public List<double> Weights { get; set; }
        public double Bias { get; set; }
        public double Delta { get; set; }

        public Node(int inputCount)
        {
            Weights = new List<double>();
            Random rand = new Random();
            double stddev = 1.0 / Math.Sqrt(inputCount); // انحراف معیار

            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(NormalRandom(rand, 0, stddev)); // مقداردهی وزن‌ها
            }

            Bias = NormalRandom(rand, 0, stddev); // مقداردهی بایاس
        }

        private double NormalRandom(Random rand, double mean, double stddev)
        {
            double u1 = 1.0 - rand.NextDouble();
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stddev * randStdNormal;
        }
    }

}

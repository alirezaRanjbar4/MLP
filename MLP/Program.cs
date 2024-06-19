using MLP;

public class Program
{
    public static void Main()
    {
        NeuralNetwork nn = new NeuralNetwork(0.1); // ایجاد شبکه عصبی با نرخ یادگیری 0.01

        nn.AddLayer(2, 2); // افزودن لایه ورودی با 2 نود
        nn.AddLayer(4, 2); // افزودن لایه مخفی با 4 نود
        nn.AddLayer(4, 4); // افزودن لایه مخفی با 4 نود
        nn.AddLayer(1, 4 ); // افزودن لایه خروجی با 1 نود

        // داده‌های آموزشی XOR
        var trainingData = new List<Tuple<List<double>, List<double>>>()
        {
            new Tuple<List<double>, List<double>>(new List<double> { 0, 0 }, new List<double> { 0 }),
            new Tuple<List<double>, List<double>>(new List<double> { 0, 1 }, new List<double> { 1 }),
            new Tuple<List<double>, List<double>>(new List<double> { 1, 0 }, new List<double> { 1 }),
            new Tuple<List<double>, List<double>>(new List<double> { 1, 1 }, new List<double> { 0 })
        };

        nn.Train(trainingData, 800000); // آموزش شبکه عصبی با داده‌های XOR برای 800,000 دوره

        // آزمایش شبکه عصبی با داده‌های XOR
        foreach (var data in trainingData)
        {
            var inputs = data.Item1;
            var targets = data.Item2;

            var outputs = nn.Predict(inputs);

            Console.WriteLine($"Input: {string.Join(", ", inputs)} - Predicted Output: {string.Join(", ", outputs)} - Expected Output: {string.Join(", ", targets)}");
        }
    }
}
namespace MLP
{
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; set; }
        public double LearningRate { get; set; }

        public NeuralNetwork(double learningRate)
        {
            Layers = new List<Layer>();
            LearningRate = learningRate;
        }

        public void AddLayer(int nodeCount, int inputCount)
        {
            Layers.Add(new Layer(nodeCount, inputCount));
        }

        public void ForwardPropagation(List<double> inputs)
        {
            // محاسبه مقدار نودهای لایه اول (لایه ورودی)0
            for (int i = 0; i < Layers[0].Nodes.Count; i++)
            {
                Layers[0].Nodes[i].Value = 0; // مقدار اولیه نود برابر صفر قرار می‌گیرد

                // محاسبه وزن‌دار ورودی‌ها
                for (int j = 0; j < inputs.Count; j++)
                {
                    // مجموع وزن‌دار ورودی‌ها را محاسبه می‌کنیم
                    Layers[0].Nodes[i].Value += inputs[j] * Layers[0].Nodes[i].Weights[j];
                }

                // اضافه کردن بایاس و اعمال تابع فعال‌سازی سیگموید
                Layers[0].Nodes[i].Value = Sigmoid(Layers[0].Nodes[i].Value + Layers[0].Nodes[i].Bias);
            }

            // محاسبه مقدار نودهای لایه‌های میانی و خروجی
            for (int i = 1; i < Layers.Count; i++)
            {
                var previousLayerOutputs = Layers[i - 1].GetOutputs(); // خروجی‌های لایه قبلی

                foreach (var node in Layers[i].Nodes)
                {
                    node.Value = 0; // مقدار اولیه نود برابر صفر قرار می‌گیرد

                    // محاسبه وزن‌دار ورودی‌های لایه قبلی
                    for (int j = 0; j < previousLayerOutputs.Count; j++)
                    {
                        // مجموع وزن‌دار ورودی‌ها را محاسبه می‌کنیم
                        node.Value += previousLayerOutputs[j] * node.Weights[j];
                    }

                    // اضافه کردن بایاس و اعمال تابع فعال‌سازی سیگموید
                    node.Value = Sigmoid(node.Value + node.Bias);
                }
            }
        }

        // مرحله پس‌رو (Backward Propagation)
        public void BackwardPropagation(List<double> targets)
        {
            // محاسبه دلتا برای لایه خروجی
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                var layer = Layers[i];

                for (int j = 0; j < layer.Nodes.Count; j++)
                {
                    var node = layer.Nodes[j];

                    if (i == Layers.Count - 1) // اگر لایه خروجی باشد
                    {
                        // محاسبه خطای خروجی: تفاضل مقدار واقعی و مقدار پیش‌بینی شده
                        // محاسبه دلتا: ضرب خطا در مشتق تابع فعال‌سازی سیگموید
                        node.Delta = (targets[j] - node.Value) * SigmoidDerivative(node.Value);
                    }
                    else // اگر لایه میانی باشد
                    {
                        node.Delta = 0.0; // مقدار اولیه دلتا برابر صفر قرار می‌گیرد

                        // جمع‌آوری دلتاهای لایه بعدی
                        for (int k = 0; k < Layers[i + 1].Nodes.Count; k++)
                        {
                            // جمع دلتاهای لایه بعدی ضرب در وزن‌های مربوطه
                            node.Delta += Layers[i + 1].Nodes[k].Weights[j] * Layers[i + 1].Nodes[k].Delta;
                        }

                        // محاسبه دلتا: ضرب مجموع دلتاهای لایه بعدی در مشتق تابع فعال‌سازی سیگموید
                        node.Delta *= SigmoidDerivative(node.Value);
                    }
                }
            }

            // به‌روزرسانی وزن‌ها و بایاس‌ها
            for (int i = 0; i < Layers.Count; i++)
            {
                // ورودی‌های لایه: اگر لایه اول باشد ورودی‌ها، در غیر این صورت خروجی‌های لایه قبلی
                var layerInputs = i == 0 ? Layers[0].Nodes.Select(n => n.Value).ToList() : Layers[i - 1].GetOutputs();

                foreach (var node in Layers[i].Nodes)
                {
                    // به‌روزرسانی وزن‌ها
                    for (int j = 0; j < layerInputs.Count; j++)
                    {
                        // به‌روزرسانی وزن: وزن قبلی به اضافه نرخ یادگیری ضرب در دلتا ضرب در ورودی لایه
                        node.Weights[j] += LearningRate * node.Delta * layerInputs[j];
                    }

                    // به‌روزرسانی بایاس: بایاس قبلی به اضافه نرخ یادگیری ضرب در دلتا
                    node.Bias += LearningRate * node.Delta;
                }
            }
        }

        // تابع آموزش شبکه عصبی
        public void Train(List<Tuple<List<double>, List<double>>> trainingData, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var data in trainingData)
                {
                    var inputs = data.Item1; // ورودی‌های داده آموزشی
                    var targets = data.Item2; // خروجی‌های هدف داده آموزشی

                    ForwardPropagation(inputs); // انجام مرحله پیش‌رو
                    BackwardPropagation(targets); // انجام مرحله پس‌رو
                }
            }
        }

        // تابع پیش‌بینی شبکه عصبی
        public List<double> Predict(List<double> inputs)
        {
            ForwardPropagation(inputs); // انجام مرحله پیش‌رو
            return Layers.Last().GetOutputs(); // بازگشت خروجی‌های لایه آخر
        }

        // تابع فعال‌سازی سیگموید
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x)); // محاسبه مقدار سیگموید
        }

        // مشتق تابع فعال‌سازی سیگموید
        private double SigmoidDerivative(double x)
        {
            return x * (1.0 - x); // محاسبه مشتق سیگموید
        }
    }
}

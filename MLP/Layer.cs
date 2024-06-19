namespace MLP
{
    public class Layer
    {
        public List<Node> Nodes { get; set; }

        public Layer(int nodeCount, int inputCount)
        {
            Nodes = new List<Node>();

            // ایجاد نودها و مقداردهی اولیه آنها
            for (int i = 0; i < nodeCount; i++)
            {
                Nodes.Add(new Node(inputCount));
            }
        }

        // دریافت خروجی‌های لایه
        public List<double> GetOutputs()
        {
            return Nodes.Select(n => n.Value).ToList();
        }
    }
}

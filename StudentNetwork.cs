using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace TopoBotCSharp
{
    public class StudentNetwork : BaseNetwork
    {
        // Настройки
        public double learning_rate = 0.15; // Скорость обучения

        // Массивы данных
        private double[][] layers;      // Значения нейронов (кэш прямого прохода)
        private double[][] errors;      // Ошибки (дельта) для обратного прохода
        private double[][,] weights;    // Веса связей

        private Random rand = new Random();
        private Stopwatch watch = new Stopwatch();

        // Классическая Сигмоида
        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        // Производная Сигмоиды: y * (1 - y)
        private double SigmoidDerivative(double y) => y * (1.0 - y);

        // Сохраним структуру явно, чтобы можно было при загрузке проверить
        private readonly int[] structure;
        public StudentNetwork(int[] structure)
        {

            this.structure = (int[])structure.Clone();

            // Инициализация массивов
            layers = new double[structure.Length][];
            errors = new double[structure.Length][];
            weights = new double[structure.Length - 1][,];

            for (int i = 0; i < structure.Length; i++)
            {
                // +1 нейрон для Bias (смещения)
                layers[i] = new double[structure[i] + 1];
                errors[i] = new double[structure[i] + 1];

                // Последний нейрон - это Bias, он всегда = 1.0
                layers[i][structure[i]] = 1.0;
            }

            // Инициализация весов
            for (int k = 0; k < structure.Length - 1; k++)
            {
                int inputs = layers[k].Length;     // Включая Bias текущего слоя
                int outputs = structure[k + 1];    // Реальные нейроны следующего (без его Bias)

                weights[k] = new double[inputs, outputs];

                for (int i = 0; i < inputs; i++)
                {
                    for (int j = 0; j < outputs; j++)
                    {
                        // Случайная инициализация. 
                        // Делим на корень из входов, чтобы при 400 входах сумма не улетала в небеса.
                        weights[k][i, j] = (rand.NextDouble() - 0.5) * (2.0 / Math.Sqrt(inputs));
                    }
                }
            }
        }

        // --- Прямой проход (с кэшированием в layers) ---
        private void ForwardPass()
        {
            for (int k = 0; k < weights.Length; k++) // По всем слоям связей
            {
                int inputs = layers[k].Length;
                int outputs = layers[k + 1].Length - 1; // Не трогаем Bias следующего слоя

                // Используем Parallel для ускорения
                Parallel.For(0, outputs, j =>
                {
                    double sum = 0;
                    for (int i = 0; i < inputs; i++)
                    {
                        sum += layers[k][i] * weights[k][i, j];
                    }
                    layers[k + 1][j] = Sigmoid(sum);
                });

                // Bias следующего слоя всегда остается 1.0, мы его не трогали
            }
        }

        // --- Обратный проход (BackPropagation) ---
        private void BackPropagation(double[] expectedOutput)
        {
            int lastLayer = layers.Length - 1;
            int outputCount = layers[lastLayer].Length - 1; // Без Bias

            // 1. Ошибка выходного слоя (MSE)
            // Error = (Target - Output) * Derivative
            Parallel.For(0, outputCount, j =>
            {
                double output = layers[lastLayer][j];
                double error = expectedOutput[j] - output;
                errors[lastLayer][j] = error * SigmoidDerivative(output);
            });

            // 2. Ошибка скрытых слоев
            for (int k = lastLayer - 1; k >= 0; k--)
            {
                int currentLayerSize = layers[k].Length;     // Включая Bias
                int nextLayerSize = layers[k + 1].Length - 1; // Реальные нейроны

                Parallel.For(0, currentLayerSize, i =>
                {
                    double sum = 0;
                    for (int j = 0; j < nextLayerSize; j++)
                    {
                        // Суммируем ошибки, пришедшие назад по весам
                        sum += errors[k + 1][j] * weights[k][i, j];
                    }
                    // Умножаем на производную текущего нейрона
                    errors[k][i] = sum * SigmoidDerivative(layers[k][i]);
                });
            }

            // 3. Обновление весов
            for (int k = 0; k < weights.Length; k++)
            {
                int inputs = layers[k].Length;
                int outputs = layers[k + 1].Length - 1;

                Parallel.For(0, outputs, j =>
                {
                    for (int i = 0; i < inputs; i++)
                    {
                        // Gradient = Error * Input
                        double gradient = errors[k + 1][j] * layers[k][i];
                        weights[k][i, j] += learning_rate * gradient;
                        // Используем += так как ошибка была (Target - Output)
                    }
                });
            }
        }

        // --- Обучение на датасете ---
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            watch.Restart();

            // Копируем список для перемешивания
            var samples = samplesSet.samples.ToList();
            double totalError = 0;

            for (int epoch = 0; epoch < epochsCount; epoch++)
            {
                // Перемешивание (Fisher-Yates) - ЭТО ВАЖНО!
                int n = samples.Count;
                while (n > 1)
                {
                    n--;
                    int k = rand.Next(n + 1);
                    var value = samples[k];
                    samples[k] = samples[n];
                    samples[n] = value;
                }

                totalError = 0;

                foreach (var sample in samples)
                {
                    // 1. Копируем вход в первый слой
                    for (int i = 0; i < sample.input.Length; i++)
                        layers[0][i] = sample.input[i];

                    // 2. Считаем
                    ForwardPass();

                    // 3. Считаем суммарную ошибку MSE для статистики
                    double sampleError = 0;
                    var lastLayer = layers[layers.Length - 1];
                    for (int i = 0; i < sample.outputVector.Length; i++)
                    {
                        sampleError += Math.Pow(sample.outputVector[i] - lastLayer[i], 2);
                    }
                    totalError += sampleError;

                    // 4. Учимся
                    BackPropagation(sample.outputVector);
                }

                totalError /= samples.Count; // Средняя ошибка по эпохе

                OnTrainProgress((double)(epoch + 1) / epochsCount, totalError, watch.Elapsed);

                if (totalError < acceptableError) break;
            }

            watch.Stop();
            return totalError;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int i = 0;
            while (i < 500)
            {
                i++;
                for (int j = 0; j < sample.input.Length; j++) layers[0][j] = sample.input[j];
                ForwardPass();

                // Проверка ошибки
                double error = 0;
                var lastLayer = layers[layers.Length - 1];
                for (int k = 0; k < sample.outputVector.Length; k++)
                    error += Math.Pow(sample.outputVector[k] - lastLayer[k], 2);

                if (error < acceptableError) break;

                BackPropagation(sample.outputVector);
            }
            return i;
        }

        protected override double[] Compute(double[] input)
        {
            // Копируем вход
            for (int i = 0; i < input.Length; i++)
                layers[0][i] = input[i];

            ForwardPass();

            // Возвращаем результат без Bias (последнего элемента)
            int outputCount = layers[layers.Length - 1].Length - 1;
            double[] result = new double[outputCount];
            Array.Copy(layers[layers.Length - 1], result, outputCount);

            return result;
        }
        // --- Сохранение / загрузка весов ---

        public override void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);

            // Сохраняем структуру
            bw.Write(structure.Length);
            foreach (int s in structure)
                bw.Write(s);

            // Сохраняем веса
            bw.Write(weights.Length);
            for (int k = 0; k < weights.Length; k++)
            {
                int inputs = weights[k].GetLength(0);
                int outputs = weights[k].GetLength(1);

                bw.Write(inputs);
                bw.Write(outputs);

                for (int i = 0; i < inputs; i++)
                {
                    for (int j = 0; j < outputs; j++)
                    {
                        bw.Write(weights[k][i, j]);
                    }
                }
            }
        }

        public override void Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("Файл весов не найден", path);

            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs);

            int len = br.ReadInt32();
            int[] fileStructure = new int[len];
            for (int i = 0; i < len; i++)
                fileStructure[i] = br.ReadInt32();

            // На простоту: проверим, что структура совпадает
            if (!fileStructure.SequenceEqual(structure))
                throw new InvalidOperationException("Структура сети в файле отличается от текущей.");

            int layersCount = br.ReadInt32();
            if (layersCount != weights.Length)
                throw new InvalidOperationException("Размерность весов в файле не совпадает.");

            for (int k = 0; k < layersCount; k++)
            {
                int inputs = br.ReadInt32();
                int outputs = br.ReadInt32();

                if (weights[k].GetLength(0) != inputs || weights[k].GetLength(1) != outputs)
                    throw new InvalidOperationException("Размерность слоя весов не совпадает.");
                for (int i = 0; i < inputs; i++)
                {
                    for (int j = 0; j < outputs; j++)
                    {
                        weights[k][i, j] = br.ReadDouble();
                    }
                }
            }
        }
    }
}

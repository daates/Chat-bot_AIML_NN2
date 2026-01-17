using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;

namespace TopoBotCSharp
{

    public enum SignType : byte { Type0 = 0, Type1, Type2, Type3, Type4, Type5, Type6, Type7, Undef };

    public class DatasetProcessor
    {
        public static string SignTypeToString(SignType type)
        {
            switch (type)
            {
                case SignType.Type0: return "apiary";
                case SignType.Type1: return "big_house";
                case SignType.Type2: return "cemetery";
                case SignType.Type3: return "church";
                case SignType.Type4: return "fir";
                case SignType.Type5: return "small_house";
                case SignType.Type6: return "tower";
                case SignType.Type7: return "yurt";
                default: return "Неизвестно";
            }
        }

        // ПУТЬ К ДАТАСЕТУ
        private const string databaseLocation = "D:\\is_4curs\\LAB8\\NeuralNetwork1\\dataset\\dataset";

        private Random random;
        public int ClassesCount { get; set; } = 8;

        private Dictionary<SignType, List<string>> structure;

        public DatasetProcessor()
        {
            random = new Random();
            structure = new Dictionary<SignType, List<string>>();

            foreach (SignType type in Enum.GetValues(typeof(SignType)))
            {
                if (type == SignType.Undef) continue;
                structure[type] = new List<string>();
            }

            foreach (var key in structure.Keys.ToList())
            {
                string folderName = SignTypeToString(key);
                string path = Path.Combine(databaseLocation, folderName);

                DirectoryInfo d = new DirectoryInfo(path);
                if (d.Exists)
                {
                    // Загружаем файлы
                    var files = d.GetFiles("*.png").Select(f => f.FullName).ToList();

                    files.Sort();

                    structure[key].AddRange(files);
                }
            }
        }


        public Bitmap ProcessAndCenterImage(Bitmap original)
        {
            int w = original.Width;
            int h = original.Height;

            int minX = w, minY = h, maxX = 0, maxY = 0;
            bool found = false;

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    var color = original.GetPixel(x, y);
                    byte r = color.R;
                    byte g = color.G;
                    byte b = color.B;

                    if ((r + g + b) / 3 < 128)
                    {
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                        found = true;
                    }
                }
            }

            if (!found) return new Bitmap(original);

            int signW = maxX - minX + 1;
            int signH = maxY - minY + 1;
            int size = Math.Max(signW, signH) + 20;

            Bitmap result = new Bitmap(size, size);
            using (Graphics gr = Graphics.FromImage(result))
            {
                gr.Clear(Color.White);
                int posX = (size - signW) / 2;
                int posY = (size - signH) / 2;
                gr.DrawImage(original, new Rectangle(posX, posY, signW, signH),
                    new Rectangle(minX, minY, signW, signH), GraphicsUnit.Pixel);
            }

            return result;
        }



        //Метод конвертации с проверкой яркости
        private double[] ImageToInputVector(Bitmap original)
        {
            // Сначала получаем центрированную картинку
            using (Bitmap centered = ProcessAndCenterImage(original))
            using (Bitmap scaled = new Bitmap(centered, new Size(20, 20)))
            {
                double[] input = new double[400];
                for (int y = 0; y < 20; y++)
                {
                    for (int x = 0; x < 20; x++)
                    {
                        double val = 1.0 - scaled.GetPixel(x, y).GetBrightness();
                        input[y * 20 + x] = (val > 0.1) ? val : 0.0;
                    }
                }
                return input;
            }
        }


        public SamplesSet getTestDataset(int count)
        {
            SamplesSet set = new SamplesSet();
            int samplesPerClass = count / ClassesCount;

            for (int i = 0; i < ClassesCount; i++)
            {
                SignType currentType = (SignType)i;
                if (!structure.ContainsKey(currentType) || structure[currentType].Count == 0) continue;

                List<string> files = structure[currentType];
                int takeCount = Math.Min(samplesPerClass, files.Count);
                int startIndex = files.Count - takeCount;

                for (int j = 0; j < takeCount; j++)
                {
                    // Берем файл с конца списка
                    string file = files[startIndex + j];

                    using (Bitmap bmp = new Bitmap(file))
                    {
                        double[] input = ImageToInputVector(bmp);
                        set.AddSample(new Sample(input, ClassesCount, currentType));
                    }
                }
            }
            set.shuffle(); // Перемешиваем уже внутри набора, чтобы тестировать вразнобой
            return set;
        }

        public SamplesSet getTrainDataset(int count)
        {
            SamplesSet set = new SamplesSet();
            int samplesPerClass = count / ClassesCount;

            for (int i = 0; i < ClassesCount; i++)
            {
                SignType currentType = (SignType)i;
                if (!structure.ContainsKey(currentType) || structure[currentType].Count == 0) continue;

                List<string> files = structure[currentType];

                // Берем с начала (индексы 0, 1, 2...)
                int takeCount = Math.Min(samplesPerClass, files.Count);

                for (int j = 0; j < takeCount; j++)
                {
                    string file = files[j];
                    using (Bitmap bmp = new Bitmap(file))
                    {
                        double[] input = ImageToInputVector(bmp);
                        set.AddSample(new Sample(input, ClassesCount, currentType));
                    }
                }
            }
            set.shuffle(); // Обязательно перемешиваем перед подачей на обучение
            return set;
        }

        public Sample getSample(Bitmap bitmap)
        {
            double[] input = ImageToInputVector(bitmap);
            return new Sample(input, ClassesCount);
        }

        public Tuple<Sample, Bitmap> getSample()
        {
            var validKeys = structure
                .Where(k => (int)k.Key < ClassesCount && k.Value.Count > 0)
                .Select(k => k.Key)
                .ToList();

            if (validKeys.Count == 0) return null;

            var type = validKeys[random.Next(validKeys.Count)];
            // Тут оставляем рандом, чтобы "играться" с разными картинками
            var file = structure[type][random.Next(structure[type].Count)];

            Bitmap bitmap = new Bitmap(file);
            double[] input = ImageToInputVector(bitmap);

            return Tuple.Create(new Sample(input, ClassesCount, type), bitmap);
        }

        public SignType RecognizeImage(string filePath, BaseNetwork network)
        {
            using (var bmp = new Bitmap(filePath))
            {
                var sample = getSample(bmp);
                var predicted = network.Predict(sample);
                return predicted;
            }
        }

        public string RecognizeImageToText(string filePath, BaseNetwork network)
        {
            var sign = RecognizeImage(filePath, network);
            if (sign == SignType.Undef)
            {
                return "Не удалось уверенно распознать знак.";
            }

            string code = SignTypeToString(sign); // apiary / big_house / ...

            // Текст, который можно либо сразу отправлять в TG, либо прокидывать в AIML через RECOGNIZED
            return $"Похоже, это знак: {code}.";
        }
    }
}

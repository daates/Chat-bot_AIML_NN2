using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AIMLbot;

namespace TopoBotCSharp
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.InputEncoding = Encoding.Unicode;
            Console.OutputEncoding = UnicodeEncoding.Unicode;

            Bot myBot = InitAimlBot();

            var datasetProcessor = new DatasetProcessor();

            int[] structure = { 400, 128, 32, 8 };
            BaseNetwork neuralNet = new StudentNetwork(structure);

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string weightsPath = Path.Combine(baseDir, "network.bin");

            if (File.Exists(weightsPath))
            {
                Console.WriteLine("Загрузка весов сети...");
                try
                {
                    neuralNet.Load(weightsPath);
                    Console.WriteLine("Веса успешно загружены.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Ошибка при загрузке весов, выполняем полное обучение: " + ex.Message);
                    TrainAndSave(datasetProcessor, neuralNet, weightsPath);
                }
            }
            else
            {
                TrainAndSave(datasetProcessor, neuralNet, weightsPath);
            }

            Console.Clear();
            Console.WriteLine("========================================");
            Console.WriteLine("БОТ-КАРТОГРАФ (Telegram) ЗАПУЩЕН!");
            Console.WriteLine("Бот тренирует/загружает сеть при старте.");
            Console.WriteLine("Открой Telegram и напиши своему боту.");
            Console.WriteLine("Нажми Enter для остановки.");
            Console.WriteLine("========================================");

            // Токен Telegram-бота (заведи его через BotFather и вставь сюда)
            string telegramToken = "8384791415:AAGI9Zqq6-mk34QM-bFbEVDxAejX5h_py5M";

            var tgHost = new TelegramHost(telegramToken, datasetProcessor, neuralNet, myBot);
            using var cts = new CancellationTokenSource();

            // Запускаем приём апдейтов
            await tgHost.RunAsync(cts.Token);

            // Ждём Enter, чтобы завершить
            Console.ReadLine();
            cts.Cancel();
        }

        private static void TrainAndSave(DatasetProcessor datasetProcessor, BaseNetwork neuralNet, string weightsPath)
        {
            Console.WriteLine("Подготовка обучающего датасета...");
            var trainSet = datasetProcessor.getTrainDataset(1040);

            Console.WriteLine("Обучение сети...");
            double finalError = neuralNet.TrainOnDataSet(
                samplesSet: trainSet,
                epochsCount: 30,
                acceptableError: 0.01,
                parallel: true);

            Console.WriteLine($"Обучение завершено. Итоговая ошибка: {finalError:F6}");

            Console.WriteLine("Сохранение весов...");
            neuralNet.Save(weightsPath);
            Console.WriteLine("Веса сохранены.");
        }

        private static Bot InitAimlBot()
        {
            Bot myBot = new Bot();

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string configDir = Path.Combine(baseDir, "config");
            string aimlDir = Path.Combine(baseDir, "aiml");
            string settingsPath = Path.Combine(configDir, "Settings.xml");

            Console.WriteLine("=== ДИАГНОСТИКА ПУТЕЙ ===");
            Console.WriteLine($"BaseDirectory: {baseDir}");
            Console.WriteLine($"Config dir:    {configDir}");
            Console.WriteLine($"AIML dir:      {aimlDir}");
            Console.WriteLine($"Settings.xml:  {settingsPath}");
            Console.WriteLine("=========================");

            if (!File.Exists(settingsPath))
            {
                Console.WriteLine("ОШИБКА: Settings.xml не найден по ожидаемому пути.");
                Console.ReadLine();
                Environment.Exit(1);
            }

            try
            {
                myBot.UpdatedConfigDirectory = configDir;
                myBot.UpdatedAimlDirectory = aimlDir;

                myBot.loadSettings(settingsPath);

                myBot.isAcceptingUserInput = false;
                myBot.loadAIMLFromFiles();
                myBot.isAcceptingUserInput = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Ошибка при инициализации бота:");
                Console.WriteLine(ex.Message);
                Console.WriteLine(ex.StackTrace);
                Console.ReadLine();
                Environment.Exit(1);
            }

            return myBot;
        }
    }
}
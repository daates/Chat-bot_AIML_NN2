using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;
using Telegram.Bot.Requests;
using AIMLbot;

namespace TopoBotCSharp
{
    public class TelegramHost
    {
        private readonly ITelegramBotClient botClient;
        private readonly DatasetProcessor datasetProcessor;
        private readonly BaseNetwork neuralNet;

        private readonly Bot aimlBot;
        private readonly AIMLbot.User aimlUser;

        private bool waitForPhoto;

        public TelegramHost(string token, DatasetProcessor processor, BaseNetwork network, Bot bot)
        {
            botClient = new TelegramBotClient(token);
            datasetProcessor = processor;
            neuralNet = network;

            aimlBot = bot;
            aimlUser = new AIMLbot.User("tg-user", aimlBot);
            waitForPhoto = false;
        }

        public async Task RunAsync(CancellationToken cancellationToken = default)
        {
            var me = await botClient.SendRequest(
                new GetMeRequest(),
                cancellationToken
            );
            Console.WriteLine($"Telegram-бот запущен: @{me.Username}");

            var receiverOptions = new ReceiverOptions
            {
                AllowedUpdates = Array.Empty<UpdateType>()
            };

            botClient.StartReceiving(
                HandleUpdateAsync,
                HandleErrorAsync,
                receiverOptions,
                cancellationToken
            );
        }

        private async Task HandleUpdateAsync(ITelegramBotClient client, Update update, CancellationToken token)
        {
            if (update.Type != UpdateType.Message || update.Message == null)
            {
                return;
            }

            var msg = update.Message;

            if (msg.Type == MessageType.Text)
            {
                await HandleTextMessage(msg, token);
            }
            else if (msg.Type == MessageType.Photo)
            {
                await HandlePhotoMessage(msg, token);
            }
        }

        private async Task HandleTextMessage(Message msg, CancellationToken token)
        {
            string text = msg.Text?.Trim() ?? string.Empty;

            if (text.Equals("/start", StringComparison.OrdinalIgnoreCase))
            {
                await botClient.SendRequest(
                    new SendMessageRequest
                    {
                        ChatId = msg.Chat.Id,
                        Text = "Привет! Я бот-картограф.\n" +
                               "Напиши \"угадать знак\", чтобы я попытался распознать знак по фото,\n" +
                               "или \"узнать инфу\", чтобы просто пообщаться и узнать о топознаках."
                    },
                    cancellationToken: token);
                return;
            }

            if (text.Equals("угадать знак", StringComparison.OrdinalIgnoreCase))
            {
                waitForPhoto = true;
                await botClient.SendRequest(
                    new SendMessageRequest
                    {
                        ChatId = msg.Chat.Id,
                        Text = "Ок, пришли мне фотографию топографического знака (как картинку)."
                    },
                    cancellationToken: token);
                return;
            }

            if (text.Equals("узнать инфу", StringComparison.OrdinalIgnoreCase))
            {
                waitForPhoto = false;
                await botClient.SendRequest(
                    new SendMessageRequest
                    {
                        ChatId = msg.Chat.Id,
                        Text = "Можем поговорить о топографических знаках. Напиши, например, \"Привет\" или \"Что ты умеешь\"."
                    },
                    cancellationToken: token);
                return;
            }

            // Режим "узнать инфу" или просто обычный текст: пробрасываем в AIML
            var request = new Request(text, aimlUser, aimlBot);
            var result = aimlBot.Chat(request);

            string answer = string.IsNullOrWhiteSpace(result.Output)
                ? "Я не совсем понял. Можешь переформулировать или написать, о каком знаке хочешь узнать?"
                : result.Output;

            await botClient.SendRequest(
                new SendMessageRequest
                {
                    ChatId = msg.Chat.Id,
                    Text = answer
                },
                cancellationToken: token);
        }

        private async Task HandlePhotoMessage(Message msg, CancellationToken token)
        {
            var chatId = msg.Chat.Id;

            try
            {
                if (!waitForPhoto)
                {
                    await botClient.SendRequest(
                        new SendMessageRequest
                        {
                            ChatId = chatId,
                            Text = "Если хочешь, чтобы я попытался угадать знак по фото, сначала напиши \"угадать знак\"."
                        },
                        cancellationToken: token);
                    return;
                }

                var photo = msg.Photo?.OrderBy(p => p.FileSize).LastOrDefault();
                if (photo == null)
                {
                    await botClient.SendRequest(
                        new SendMessageRequest
                        {
                            ChatId = chatId,
                            Text = "Не вижу фото в сообщении."
                        },
                        cancellationToken: token);
                    return;
                }

                // 1. Получаем файл с сервера Telegram
                var file = await botClient.SendRequest(
                    new GetFileRequest
                    {
                        FileId = photo.FileId
                    },
                    cancellationToken: token);

                // 2. Сохраняем локально
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string tempDir = Path.Combine(baseDir, "temp");
                Directory.CreateDirectory(tempDir);

                string localPath = Path.Combine(tempDir, $"{file.FileId}.jpg");

                string fileUrl = $"https://api.telegram.org/file/bot{((TelegramBotClient)botClient).Token}/{file.FilePath}";

                using (var http = new System.Net.Http.HttpClient())
                await using (var fs = new FileStream(localPath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    using var response = await http.GetAsync(fileUrl, token);
                    response.EnsureSuccessStatusCode();
                    await response.Content.CopyToAsync(fs, token);
                }

                // 3. Распознаём знак нейросетью
                var sign = datasetProcessor.RecognizeImage(localPath, neuralNet);
                string code = DatasetProcessor.SignTypeToString(sign);

                string answer;

                if (sign == SignType.Undef || code == "Неизвестно")
                {
                    answer = "Я не смог уверенно распознать этот знак.";
                }
                else
                {
                    answer = $"Похоже, это знак: {code}.";
                }

                // После одного угадывания можно сбросить режим
                waitForPhoto = false;

                // 4. Отправляем ответ пользователю
                await botClient.SendRequest(
                    new SendMessageRequest
                    {
                        ChatId = chatId,
                        Text = answer
                    },
                    cancellationToken: token);

                // 5. Чистим временный файл
                try
                {
                    File.Delete(localPath);
                }
                catch
                {
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка TG при обработке фото: {ex}");
                await botClient.SendRequest(
                    new SendMessageRequest
                    {
                        ChatId = chatId,
                        Text = "Произошла ошибка при обработке изображения."
                    },
                    cancellationToken: token);
            }
        }

        private Task HandleErrorAsync(ITelegramBotClient client, Exception exception, CancellationToken token)
        {
            var errorMessage = exception switch
            {
                ApiRequestException apiEx => $"Telegram API Error:\n[{apiEx.ErrorCode}] {apiEx.Message}",
                _ => exception.ToString()
            };

            Console.WriteLine(errorMessage);
            return Task.CompletedTask;
        }
    }
}
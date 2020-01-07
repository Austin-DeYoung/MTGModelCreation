using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace MTGModelCreation
{
    class Program
    {
        static readonly string trainingData = @"C:\Users\austi\OneDrive\Desktop\MachineLearning\FinalProject\MTGModelCreation\MTGModelCreation\TrainingData.csv";
        static readonly string testData = @"C:\Users\austi\OneDrive\Desktop\MachineLearning\FinalProject\MTGModelCreation\MTGModelCreation\TestingData.csv";
        //static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");


        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, trainingData);
            
            Evaluate(mlContext, model);

            TestSinglePredition(mlContext, model);

            Console.ReadKey();
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<CardData>(dataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Price")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ArtistEncoded", inputColumnName: "Artist"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "LoyaltyEncoded", inputColumnName: "Loyalty"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ManaCostEncoded", inputColumnName: "ManaCost"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "NameEncoded", inputColumnName: "Name"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PowerEncoded", inputColumnName: "Power"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RarityEncoded", inputColumnName: "Rarity"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TextEncoded", inputColumnName: "Text"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ToughnessEncoded", inputColumnName: "Toughness"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TypeEncoded", inputColumnName: "Type"))
                .Append(mlContext.Transforms.Concatenate("Features", "ArtistEncoded", "ConvertedManaCost", "LoyaltyEncoded", "ManaCostEncoded", "NameEncoded", "PowerEncoded", "RarityEncoded", "TextEncoded", "ToughnessEncoded", "TypeEncoded"))
                .Append(mlContext.Regression.Trainers.Sdca());
                //Attempted to use FastTree method, but recieved errors
                
            var model = pipeline.Fit(dataView);
            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<CardData>(testData, hasHeader: true, separatorChar: ',');

            var predictions = model.Transform(dataView);

            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine("RSquared Score: {0}", metrics.RSquared);
            Console.WriteLine("Root Mean Squared Error: {0}", metrics.RootMeanSquaredError);
            //Console.WriteLine("RSquared Score: {metrics.RSquared:0.##}");
            //Console.WriteLine("Root Mean Squared Error: {metrics.RootMeanSquaredError:#.##}");
        }

        private static void TestSinglePredition(MLContext mLContext, ITransformer model)
        {
            var predictFunction = mLContext.Model.CreatePredictionEngine<CardData, PricePrediction>(model);

            var lightningStrike = new CardData()
            {
                Artist = "",
                ConvertedManaCost = 4,
                Loyalty = "",
                ManaCost = "{3}{G}",
                Name = "",
                Power = "2",
                Rarity = "uncommon",
                Text = "",
                Toughness = "3",
                Type = "Creature",
                //Price = 0 //Actual = 0.24
            };

            var vivienReid = new CardData()
            {
                ConvertedManaCost = 5,
                Loyalty = "5",
                ManaCost = "{3}{G}{G}",
                //Name = "Huatli",
                Rarity = "mythic",
                Type = "Legendary Planeswalker",
                //Price = 0 //Actual = 9.32
            };

            var prediction = predictFunction.Predict(lightningStrike);

            Console.WriteLine();
            //Console.WriteLine("Predicted Price: {0}, Actual Price: 0.24", prediction.Price);
            Console.WriteLine("Predicted Price: {0}, Actual Price: 0.24", Math.Round(Convert.ToDouble(prediction.Price),2));
        }
    }
}

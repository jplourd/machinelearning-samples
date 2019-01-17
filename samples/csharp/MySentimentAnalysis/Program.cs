using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Transforms.Text;

namespace MySentimentAnalysis
{
    class Program
    {
        // Data file header and coupla lines
        // Sentiment SentimentText
        // 1	  ==RUDE== Dude, you are rude upload that carl picture back, or else.
        // 1	  == OK! ==  IM GOING TO VANDALIZE WILD ONES WIKI THEN!!!   
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static TextLoader _textLoader;

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);
            _textLoader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label",DataKind.Bool,0),
                    new TextLoader.Column("SentimentText",DataKind.Text,1)
                }
            });

            var model = Train(mlContext, _trainDataPath);

            //Loads the test dataset.
            //Creates the binary evaluator.
            //Evaluates the model and create metrics.
            //Displays the metrics.
            //Save Model to external file
            Evaluate(mlContext, model);

            Predict(mlContext, model);

            PredictWithModelLoadedFromFile(mlContext);

            Console.WriteLine();
            Console.WriteLine("=============== End of process ===============");
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            // As the input and output of Transforms, a DataView is the fundamental data pipeline type,
            //comparable to IEnumerable for LINQ
            IDataView dataView = _textLoader.Read(dataPath);

            // ML.NET's transform pipelines compose a custom set of transforms that are applied to your data before training or testing. The transforms' primary purpose is
            //data featurization. Machine learning algorithms understand featurized data, so the next step is to transform our textual data into a format that our ML
            //algorithms recognize.That format is a numeric vector.

            //Next, call mlContext.Transforms.Text.FeaturizeText which featurizes the text column(SentimentText) column into a numeric vector
            //called Features used by the machine learning algorithm.This is a wrapper call that returns an EstimatorChain<TLastTransformer> that will
            //effectively be a pipeline.Name this pipeline as you will then append the trainer to the EstimatorChain.

            var pipeline = mlContext.Transforms.Text.FeaturizeText("SentimentText", "Features")

           // To add the trainer, call the mlContext.Transforms.Text.FeaturizeText wrapper method which returns a FastTreeBinaryClassificationTrainer object.This is
           //a decision tree learner you'll use in this pipeline. The FastTreeBinaryClassificationTrainer is appended to the pipeline and accepts the 
           //featurized SentimentText(Features) and the Label input parameters to learn from the historic data.
           .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            //          You train the model, TransformerChain< TLastTransformer >, based on the dataset that has been loaded and transformed. Once the estimator has been
            //defined, you train your model using the Fitwhile providing the already loaded training data. This returns a model to use for
            //predictions.pipeline.Fit() trains the pipeline and returns a Transformer based on the DataView passed in. The experiment is not executed until this happens
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = pipeline.Fit(dataView);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            // you have a model of type TransformerChain<TLastTransformer> that can be integrated into any of your .NET applications.
            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath);

            Console.WriteLine("========================== Evaluating Model  accuracy with Test data =======");
            var predictions = model.Transform(dataView);

            // The BinaryClassificationContext.Evaluate method computes the quality metrics for
            // the PredictionModel using the specified dataset.It returns a
            //BinaryClassificationEvaluator.CalibratedResult object containing the overall
            //metrics computed by binary classification evaluators. To display these to
            //determine the quality of the model, you need to get the metrics first. 
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            SaveModelAsFile(mlContext, model);

        }

        private static void Predict(MLContext mlContext, ITransformer model)
        {
            //Creates a single comment of test data.
            //Predicts sentiment based on test data.
            //Combines test data and predictions for reporting.
            //Displays the predicted results.

            //While the model is a transformer that operates on many rows of data, a very
            //common production scenario is a need for predictions on individual examples.
            // The PredictionFunction<TSrc, TDst> is a wrapper that is returned from the
            //MakePredictionFunction method.
            var predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);

            SentimentData sampleStatement = new SentimentData
            { SentimentText = "This is a very rude movie" };

            //To get a prediction, use Predict(TSrc) on the data.Note that the input data i
            //is a string and the model includes the featurization.Your pipeline is in sync
            //during training and prediction.You didn’t have to write
            //preprocessing / featurization code specifically for predictions, and the
            //  same API takes care of both batch and one - time predictions.
            var resultprediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {resultprediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public static void PredictWithModelLoadedFromFile(MLContext mlContext)
        {
            //Creates batch test data.
            //Predicts sentiment based on test data.
            //Combines test data and predictions for reporting.
            //Displays the predicted results.
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {SentimentText = "This is a very rude movie" },
                new SentimentData
                {SentimentText= "He is the best, and the article should say that." }
            };

            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            //Now that you have a model, you can use that to predict the Toxic or Non Toxic
            //sentiment of the comment data using the Transform(IDataView) method.To get a
            //prediction, use Predict on new data.Note that the input data is a string
            //and the model includes the featurization.Your pipeline is in sync during
            //training and prediction.You didn’t have to write preprocessing / featurization
            //code specifically for predictions, and the same API takes care of both batch
            //and one - time predictions.

            var sentimentStreamingDatView = mlContext.CreateStreamingDataView(sentiments);
            var predictions = loadedModel.Transform(sentimentStreamingDatView);

            // Use the model to predict whether comment data is toxic (1) or nice (0).
            var predictedResults = predictions.AsEnumerable<SentimentPrediction>(mlContext, reuseRowObject: false);

            //combine the sentiment and prediction together to see the original comment with its predicted sentiment.
            //The following code uses the Zip method to make that happen,
            var sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));
            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {item.prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");

            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");

        }
        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            //create a method to save the model so that it can be reused and consumed in other
            // applications.The ITransformer has a SaveTo(IHostEnvironment, Stream) method that takes 
            //in the _modelPath global field, and a Stream. To save this as a zip
            //file, you'll create the FileStream immediately before calling the SaveTo method.
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}

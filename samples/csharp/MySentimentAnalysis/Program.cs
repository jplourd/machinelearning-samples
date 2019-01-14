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

            Console.WriteLine("Hello World!");
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

            var pipeline = mlContext.Transforms.Text.FeaturizeText("SentimentText", "features")

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
    }
}

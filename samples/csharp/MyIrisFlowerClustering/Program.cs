// <SnippetUsingsForPaths>
using System;
using System.IO;
// </SnippetUsingsForPaths>

// <SnippetMLUsings>
using Microsoft.ML;
using Microsoft.ML.Data;
// </SnippetMLUsings>

namespace IrisFlowerClustering
{
    class Program
    {
        // <SnippetPaths>
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");
        // </SnippetPaths>

        static void Main(string[] args)
        {
            // <SnippetCreateContext>
            var mlContext = new MLContext(seed: 0);
            // </SnippetCreateContext>

            // <SnippetSetupTextLoader>
            TextLoader textLoader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = false,
                Column = new[]
                            {
                                new TextLoader.Column("SepalLength", DataKind.R4, 0),
                                new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                                new TextLoader.Column("PetalLength", DataKind.R4, 2),
                                new TextLoader.Column("PetalWidth", DataKind.R4, 3)
                            }
            });
            // </SnippetSetupTextLoader>

            // <SnippetCreateDataView>
            //The iris.data file contains five columns that represent:
            //sepal length in centimetres
            //sepal width in centimetres
            //petal length in centimetres
            //petal width in centimetres
            //type of iris flower
            IDataView dataView = textLoader.Read(_dataPath);
            // </SnippetCreateDataView>

            // <SnippetCreatePipeline>
            string featuresColumnName = "Features";
            //use a KMeansPlusPlusTrainer trainer to train the model using the k-means++ clustering algorithm.
            //With K-means, the data is clustered into a specified number of clusters in order to minimize the within-cluster sum of squares.
            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, clustersCount: 3));
            // </SnippetCreatePipeline>

            // <SnippetTrainModel>
            var model = pipeline.Fit(dataView);
            // </SnippetTrainModel>

            // <SnippetSaveModel>
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fileStream);
            }
            // </SnippetSaveModel>

            // <SnippetPredictor>
            var predictor = model.CreatePredictionEngine<IrisData, ClusterPrediction>(mlContext);
            // </SnippetPredictor>

            // <SnippetPredictionExample>
            //Determine cluster the test static data Setosa belongs to
            var prediction = predictor.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
            Console.WriteLine("***** Complete *****");
            // </SnippetPredictionExample>
        }
    }
}

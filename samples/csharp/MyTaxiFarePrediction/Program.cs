// <Snippet1>
using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Normalizers;
using Microsoft.ML.Transforms.Text;
// </Snippet1>

namespace TaxiFarePrediction
{
    //You want to predict the price value, which is a real value, based on the other factors in the data set. 
    //To do that you choose a regression machine learning task.

    class Program
    {
        // <Snippet2>
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static TextLoader _textLoader;
        // </Snippet2>

        static void Main(string[] args)
        {
            Console.WriteLine(Environment.CurrentDirectory);

            MLContext mlContext = new MLContext(seed: 0);

            _textLoader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                            {
                                new TextLoader.Column("VendorId", DataKind.Text, 0),
                                new TextLoader.Column("RateCode", DataKind.Text, 1),
                                new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                                new TextLoader.Column("TripTime", DataKind.R4, 3),
                                new TextLoader.Column("TripDistance", DataKind.R4, 4),
                                new TextLoader.Column("PaymentType", DataKind.Text, 5),
                                new TextLoader.Column("FareAmount", DataKind.R4, 6)
                            }
            }
            );
            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext);

            Console.WriteLine("*********  End **********");
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            // Loads the data.
            //Extracts and transforms the data.
            //Trains the model.
            //Saves the model as .zip file.
            //Returns the model.
            IDataView dataView = _textLoader.Read(dataPath);

            //When the model is trained and evaluated, by default, the values in the Label column are considered as correct values to be predicted. 
            //As we want to predict the taxi trip fare, copy the FareAmount column into the Label column. To do that, 
            //use the CopyColumnsEstimator transformation class
            var pipeline = mlContext.Transforms.CopyColumns("FareAmount", "Label")
                    //The algorithm that trains the model requires numeric features, so you have to transform the categorical data 
                    //(VendorId, RateCode, and PaymentType) values into numbers. To do that, use the OneHotEncodingEstimator transformation 
                    // class, which assigns different numeric key values to the different values in each of the columns,
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))
                     // The last step in data preparation combines all of the feature columns into the Features column using the 
                     // ColumnConcatenatingEstimator transformation class. By default, a learning algorithm processes 
                     // only features from the Features column
                     .Append(mlContext.Transforms.Concatenate("Features", "VendorId", "RateCode", "PassengerCount", "TripTime", "TripDistance", "PaymentType"))
                        // select a learning algorithm (learner). The learner trains the model. We chose a regression task for this problem, 
                        // so we use a FastTreeRegressionTrainer learner, which is one of the regression learners provided by ML.NET.

                        // The FastTreeRegressionTrainer learner utilizes gradient boosting. Gradient boosting is a machine learning technique 
                        // for regression problems. It builds each regression tree in a step - wise fashion.It uses a pre - defined loss function 
                        // to measure the error in each step and correct for it in the next.The result is a prediction model that is 
                        // actually an ensemble of weaker prediction models
                        .Append(mlContext.Regression.Trainers.FastTree());


            Console.WriteLine("=============== Create and Train the Model ===============");

            var model = pipeline.Fit(dataView);

            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            SaveModelAsFile(mlContext, model);
            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            // Loads the test dataset.
            //Creates the regression evaluator.
            //Evaluates the model and creates metrics.
            //Displays the metrics.
            IDataView dataView = _textLoader.Read(_testDataPath);
            //use the machine learning model parameter (a transformer) to input the features and return predictions
            var predictions = model.Transform(dataView);

            //computes the quality metrics for the PredictionModel using the specified dataset. 
            //It returns a RegressionMetrics object that contains the overall metrics computed by regression evaluators.
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            //RSquared is another evaluation metric of the regression models. 
            // RSquared takes values between 0 and 1. The closer its value is to 1, the better the model is
            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");

            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");

            Console.WriteLine($"*************************************************");

        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            //Creates a single comment of test data.
            //Predicts fare amount based on test data.
            //Combines test data and predictions for reporting.
            //Displays the predicted results.

            //load the model
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            //Prediction test
            // Create prediction function and make prediction.
            //The PredictionEngine<TSrc,TDst> is a wrapper that is returned from the CreatePredictionEngine method to test on single value
            var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);
            //Sample: 
            //vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
            //VTS,1,1,1140,3.75,CRD,15.5
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CSH", //"CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };
            var prediction = predictionFunction.Predict(taxiTripSample);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fileStream);

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}

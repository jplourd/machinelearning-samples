﻿using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace MySentimentAnalysis
{
    //SentimentData is the input dataset class and has a float (Sentiment) that has a value for sentiment of either positive or negative, and a string for the comment
    //(SentimentText). Both fields have Column attributes attached to them.This attribute describes the order of each field in the data file, and which is the Label field.
    //SentimentPrediction is the class used for prediction after the model has been trained.It has a single boolean(Sentiment) and a PredictedLabel ColumnName attribute.The
    //Label is used to create and train the model, and it's also used with a second dataset to evaluate the model. The PredictedLabel is used during prediction and evaluation. For 
    //evaluation, an input with training data, the predicted values, and the model are used.


    public class SentimentData
    {
        [Column(ordinal: "0", name: "Label")]
        public float Sentiment;
        [Column(ordinal: "1")]
        public string SentimentText;
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }
}

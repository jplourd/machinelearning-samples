// <Snippet1>
using Microsoft.ML.Data;
// </Snippet1>

namespace TaxiFarePrediction
{
    //TaxiTrip is the input data class and has definitions for each of the data seti
    // columns.Use the ColumnAttribute attribute to specify the indices of the
    // source columns in the data set.

    //The TaxiTripFarePrediction class represents predicted results.It has a single
    // float field, FareAmount, with a Score ColumnNameAttribute attribute applied.
    // In case of the regression task the Score column contains predicted label values.

    public class TaxiTrip
    {
        [Column("0")]
        public string VendorId;

        [Column("1")]
        public string RateCode;

        [Column("2")]
        public float PassengerCount;

        [Column("3")]
        public float TripTime;

        [Column("4")]
        public float TripDistance;

        [Column("5")]
        public string PaymentType;

        [Column("6")]
        public float FareAmount;
    }

    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
    // </Snippet2>
}

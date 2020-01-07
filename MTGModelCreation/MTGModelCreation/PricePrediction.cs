using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MTGModelCreation
{
    class PricePrediction
    {
        [ColumnName("Score")]
        public float Price;
    }
}
